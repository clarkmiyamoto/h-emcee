from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from .base import Adapter

class ChEESState(NamedTuple):
    """
    Args:
        T (float): Current integration time
    """
    log_T: float
    log_T_bar: float
    first_moment: float
    second_moment: float
    iteration: int 
    halton: float

class ChEESParameters(NamedTuple):
    """
    Parameters for the ChEES adaptation.
    
    Args:
        T_min: Minimum allowed integration time.
        T_max: Maximum allowed integration time.
        T_interpolation: Running average interpolation (0.9 means 90% old, 10% new)
        jitter: Strength of jittered time t_n = h_n T (where h_n is a Halton sequence) 

        lr_T: Learning rate for the integration time.
        beta1: Beta1 for ADAM optimizer.
        beta2: Beta2 for ADAM optimizer.
        regularization: Regularization for ADAM optimizer.
    """
    T_min: float = 0.25
    T_max: float = 10.0
    T_interpolation: float = 0.9
    jitter: float = 0.6

    # ADAM Optimizer Parameters
    lr_T: float = 0.025
    beta1: float = 0.0
    beta2: float = 0.95
    regularization: float = 1e-7

class ChEESAdapter(Adapter):
    """ChEES adapter for integration time adaptation."""
    
    def __init__(self, parameters: ChEESParameters, move_type: str, initial_step_size: float, initial_L: float):
        self.parameters = parameters
        self.passthrough_step_size = initial_step_size
        self.initial_L = initial_L

        self.move_type = move_type

        if self.move_type == 'hmc_move':
            self.innerproduct = lambda group1, group2, projected_momentum1: jnp.einsum('bd,bd->b', group1, projected_momentum1)
        elif (self.move_type == 'hmc_walk_move') or (self.move_type == 'hmc_side_move'):
            def innerproduct(group1: jnp.ndarray, 
                             group2: jnp.ndarray, 
                             projected_momentum1: jnp.ndarray) -> float:
                '''
                Args:
                    group1: Shape (walkers_group1, dim)
                    group2: Shape (walkers_group2, dim)
                    projected_momentum1: Shape (walkers_group1, dim)

                Returns:
                    Inner product of group1 and projected_momentum1 with 
                        respect to the covariance of group2. Shape (walkers_group1,)
                '''
                covariance = jnp.cov(group2, rowvar=False)
                covariance = jnp.atleast_2d(covariance)  # Ensure it's at least 2D
                result = jnp.linalg.solve(covariance, projected_momentum1.T).T
                return jnp.sum(group1 * result, axis=1)  # Shape: (walkers_group1,)
            self.innerproduct = innerproduct
        else:
            raise ValueError('Move type for ChEES tuner is not suppported. Choose between ["hmc", "hmc_walk", "hmc_side"]')
    
    def init(self, dim: int) -> ChEESState:
        """Initialize ChEES state."""
        return ChEESState(
            log_T = jnp.log(self.initial_L * self.passthrough_step_size),
            log_T_bar = jnp.log(self.initial_L * self.passthrough_step_size),
            first_moment = 0.0,
            second_moment = 0.0,
            iteration = 1.0,
            halton = get_halton(1.0),
        )
    
    def update(self, 
               state: ChEESState, 
               log_accept_rate: jnp.ndarray, 
               position_current: jnp.ndarray,
               position_proposed: jnp.ndarray,
               momentum_proposed: jnp.ndarray,
               group2: jnp.ndarray) -> ChEESState:
        """
        Update integration time using ChEES.
        
        Args:
            state: State of ChEES algorithm
            log_accept_rate: Shape (num_walkers,)
            position_current: Shape (num_walkers, dim)
            position_proposed: Shape (num_walkers, dim)
            momentum_proposed: Shape (num_walkers, dim)
            group2: Shape (walkers_group2, dim)
            jitter: .
                Meaning you ran leapfrog for $L = round(t_n / step_size)$ steps
        """    
        acceptance_prob = jnp.clip(jnp.exp(log_accept_rate), 0.0, 1.0) # Shape (num_walkers,)
        
        # ChEES Criterion
        mean_previous     = jnp.mean(position_current, axis=0)
        mean_proposed     = jnp.mean(position_proposed, axis=0)
        centered_previous = position_current - mean_previous
        centered_proposed = position_proposed - mean_proposed

        # Compute squared norm difference per walker
        diff_sqnorm = (
            jnp.sum(centered_proposed**2, axis=1)
            - jnp.sum(centered_previous**2, axis=1)
        )  # (num_walkers,)

        # Gradient of ChEES Criterion
        T   = jnp.exp(state.log_T)
        t_n = state.halton * T
        g_m = t_n * diff_sqnorm * self.innerproduct(centered_proposed, group2, momentum_proposed)
        # Filter out gradients from extreme cases
        g_m = jnp.where(acceptance_prob > 1e-4, g_m, 0.0)
        g_m = jnp.where(jnp.isfinite(g_m), g_m, 0.0)
        # Weight gradient signal by acceptance probability
        g_hat = jnp.sum(acceptance_prob * g_m) / (jnp.sum(acceptance_prob) + self.parameters.regularization) # Shape (1,)

        # ADAM Optimization
        iteration = state.iteration + 1
    
        first_moment       = self.parameters.beta1 * state.first_moment + (1 - self.parameters.beta1) * g_hat
        second_moment      = self.parameters.beta2 * state.second_moment + (1 - self.parameters.beta2) * g_hat ** 2
        bias_corrected_m_t = first_moment / (1 - self.parameters.beta1 ** iteration)
        bias_corrected_v_t = second_moment / (1 - self.parameters.beta2 ** iteration)

        # Gradient ASCENT (note sign, yes I lost a lot of time on this...)
        update = self.parameters.lr_T * bias_corrected_m_t / jnp.sqrt(bias_corrected_v_t + self.parameters.regularization)
        log_T = state.log_T + update # jnp.clip(update, -0.35, 0.35) 

        # Clipping log_T before computing running average
        log_T = jnp.clip(log_T, jnp.log(self.parameters.T_min), jnp.log(self.parameters.T_max))
        
        alpha = self.parameters.T_interpolation
        log_T_bar = jnp.logaddexp(
            jnp.log(alpha) + state.log_T_bar,
            jnp.log(1.0 - alpha) + log_T
        )
        
        # Clipping final results (already clipped log_T, but clip log_T_bar too)
        log_T_bar = jnp.clip(log_T_bar, jnp.log(self.parameters.T_min), jnp.log(self.parameters.T_max))

        # Report final results
        halton = get_halton(iteration)
        return ChEESState(
            log_T=log_T, 
            log_T_bar=log_T_bar, 
            first_moment=first_moment, 
            second_moment=second_moment, 
            iteration=iteration, 
            halton=halton,
        )
    
    def value(self, state: ChEESState) -> tuple[float, int]:
        """Get current integration length. Returns (unchanged_step_size, integration_length)."""
        T = self._get_T(state)
        integration_length = jnp.maximum(jnp.ceil(T/self.passthrough_step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (self.passthrough_step_size, integration_length)
    
    def finalize(self, state: ChEESState) -> tuple[float, int]:
        """Get final integration length. Returns (unchanged_step_size, integration_length)."""
        T = self._get_T(state, bar=True)
        integration_length = jnp.maximum(jnp.ceil(T/self.passthrough_step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (self.passthrough_step_size, integration_length)
    
    def _get_T(self, state: ChEESState, bar=False):
        '''
        Get's integration time w/ jittering
        '''
        log_T = state.log_T_bar if bar else state.log_T
        T = jnp.exp(log_T)
        jitter_amount = state.halton 
        T_jitter = (1.0 - self.parameters.jitter) * T + self.parameters.jitter * jitter_amount * T
        return T_jitter

@jax.jit
def get_halton(index: int, base: int = 2) -> jnp.ndarray:
    """1D Halton value at 1-based index, JAX-compatible.

    Args:
        index: 1-based integer index (>= 1)
        base:  integer base (usually 2 for 1D)
    """
    i0 = jnp.asarray(index, dtype=jnp.int32)
    b  = jnp.asarray(base,  dtype=jnp.int32)

    f0 = jnp.array(1.0, dtype=jnp.float32)
    r0 = jnp.array(0.0, dtype=jnp.float32)

    def cond_fun(state):
        i, f, r = state
        return i > 0

    def body_fun(state):
        i, f, r = state
        f = f / jnp.float32(b)
        digit = jnp.mod(i, b)
        r = r + f * digit
        i = i // b
        return (i, f, r)

    _, _, r = jax.lax.while_loop(cond_fun, body_fun, (i0, f0, r0))
    return r  # in (0, 1)