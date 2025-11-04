from typing import NamedTuple, Callable
import jax.numpy as jnp
from .base import Adapter

class ChEESState(NamedTuple):
    """
    Args:
        T (float): Current integration time
    """
    log_T: float
    first_moment: float
    second_moment: float
    iteration: int 

class ChEESParameters(NamedTuple):
    """
    Parameters for the ChEES adaptation.
    
    Args:
        T_min: Minimum allowed integration time.
        T_max: Maximum allowed integration time.

        lr_T: Learning rate for the integration time.
        beta1: Beta1 for ADAM optimizer.
        beta2: Beta2 for ADAM optimizer.
        epsilon: Epsilon for ADAM optimizer.
    """
    T_min: float = 0.25
    T_max: float = 10.0
    T_interpolation: float = 0.9

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
            first_moment = 0.0,
            second_moment = 0.0,
            iteration = 0
        )
    
    def update(self, 
               state: ChEESState, 
               log_accept_rate: jnp.ndarray, 
               position_current: jnp.ndarray,
               position_proposed: jnp.ndarray,
               momentum_proposed: jnp.ndarray,
               group2: jnp.ndarray,
               jitter: float) -> ChEESState:
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
        acceptance_prob = jnp.clip(jnp.exp(log_accept_rate), 0.0, 1.0) # Shape (batch_size,)

        mean_previous = jnp.mean(position_current, axis=0)
        mean_proposed = jnp.mean(position_proposed, axis=0)

        centered_previous = position_current - mean_previous
        centered_proposed = position_proposed - mean_proposed

        diff_sqnorm = jnp.sum(centered_proposed**2, axis=1) - jnp.sum(centered_previous**2, axis=1) # Shape ()

        # Computes gradient signal according to O(n) invariant inner product or by affine invariant inner product
        g_m = jitter * diff_sqnorm * self.innerproduct(centered_proposed, group2, momentum_proposed)
        # Filter out gradients from extreme cases
        g_m = jnp.where(acceptance_prob > 1e-4, g_m, 0.0)
        g_m = jnp.where(jnp.isfinite(g_m), g_m, 0.0)

        # Weight gradient signal by acceptance probability
        g_hat = jnp.sum(acceptance_prob * g_m) / jnp.sum(acceptance_prob + self.parameters.regularization) # Shape (1,)

        state_proposed = adam_optimizer_step(g_hat, state, self.parameters)
    
        return state_proposed
    
    def value(self, state: ChEESState) -> tuple[float, int]:
        """Get current integration length. Returns (unchanged_step_size, integration_length)."""
        T = jnp.exp(state.log_T)
        integration_length = jnp.maximum(jnp.round(T/self.passthrough_step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (self.passthrough_step_size, integration_length)
    
    def finalize(self, state: ChEESState) -> tuple[float, int]:
        """Get final integration length. Returns (unchanged_step_size, integration_length)."""
        T = jnp.exp(state.log_T)
        integration_length = jnp.maximum(jnp.round(T/self.passthrough_step_size), 1)
        integration_length = jnp.astype(integration_length, int)
        return (self.passthrough_step_size, integration_length)
    
    def _get_T(self, state):
        return jnp.exp(state.log_T)


def adam_optimizer_step(gradient_signal: float, 
                        state: ChEESState,
                        parameters: ChEESParameters) -> ChEESState:
    '''
    Performs single step of ADAM optimzier
    
    Args:
        gradient_signal: Gradient signal.
        state: State of ChEES algorithm.
        parameters: Parameters for ADAM optimizer.
    '''
    iteration = state.iteration + 1
    
    first_moment = parameters.beta1 * state.first_moment + (1 - parameters.beta1) * gradient_signal
    second_moment = parameters.beta2 * state.second_moment + (1 - parameters.beta2) * gradient_signal ** 2

    bias_corrected_m_t = first_moment / (1 - parameters.beta1 ** iteration)
    bias_corrected_v_t = second_moment / (1 - parameters.beta2 ** iteration)

    log_T_update = state.log_T - parameters.lr_T * bias_corrected_m_t / jnp.sqrt(bias_corrected_v_t + parameters.regularization)
    log_T_update = jnp.clip(log_T_update, min=-0.35, max=0.35)
    log_T = state.log_T + log_T_update

    log_T = jnp.logaddexp(log_T + jnp.log(parameters.T_interpolation), state.log_T + jnp.log(1 - parameters.T_interpolation))
    log_T = jnp.log(jnp.clip(jnp.exp(log_T), min=parameters.T_min, max=parameters.T_max))

    return ChEESState(log_T=log_T, first_moment=first_moment, second_moment=second_moment, iteration=iteration)