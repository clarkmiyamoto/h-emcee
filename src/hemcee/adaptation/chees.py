from typing import NamedTuple
import jax.numpy as jnp
from .base import Adapter

class ChEESState(NamedTuple):
    """
    Args:
        q (jnp.ndarray): Positions of walkers. Shape (batch_size, dim)
        mu (jnp.ndarray). Shape (dim).
        T (float): Current integration time
    """
    q: jnp.ndarray           # (B,d)
    mu: jnp.ndarray          # (d,)
    T: float

class ChEESParameters(NamedTuple):
    """
    Parameters for the ChEES adaptation.
    
    Args:
        T_min: Minimum allowed integration time.
        T_max: Maximum allowed integration time.
        lr_T: Learning rate for the integration time.
        mu_momentum: Momentum for the running mean.
        jitter: Jitter for the integration time.
        metric: Use O(n) invariant metric, or affine invariant metric. Choices ('euclidean', 'mahalanobis')
    """
    T_min: float = 0.25
    T_max: float = 10.0
    lr_T: float = 5e-3
    mu_momentum: float = 0.9  # EMA momentum for running mean
    jitter: float = 0.01 
    metric: str = 'euclidean' # or 'mahalanobis' 


class ChEESAdapter(Adapter):
    """ChEES adapter for integration time adaptation."""
    
    def __init__(self, parameters: ChEESParameters, initial_step_size: float, initial_L: float):
        self.parameters = parameters
        self.passthrough_step_size = initial_step_size
        self.inital_L = initial_L

        if parameters.metric == 'euclidean':
            self.chess_grad = chees_grad_T_euclidean
        elif parameters.metric == 'mahalanobis':
            self.chess_grad = chees_grad_T_mahalanobis
        else:
            raise ValueError('Metric for ChEES tuner is not suppported. Choose between ["euclidean", "mahalanobis"]')
    
    def init(self, dim: int) -> ChEESState:
        """Initialize ChEES state."""
        return ChEESState(
            q=jnp.zeros(dim),
            mu=jnp.zeros(dim),
            T=self.inital_L
        )
    
    def update(self, state: ChEESState, accept_rate: float, positions: jnp.ndarray) -> ChEESState:
        """Update integration time using ChEES."""
        # Update running mean across batch
        mu_new = welford_ema_mean(state.mu, positions, self.parameters.mu_momentum)
        
        # Simplified ChEES gradient (positions-only version)
        dq0 = state.q - state.mu
        dq1 = jnp.mean(positions, axis=0) - state.mu
        qnorm_diff = jnp.sum(dq1 * dq1) - jnp.sum(dq0 * dq0)
        grad_T = accept_rate * qnorm_diff * state.T
        
        # Gradient ascent on log T
        logT = jnp.log(state.T) + self.parameters.lr_T * grad_T
        T_new = jnp.clip(jnp.exp(logT), self.parameters.T_min, self.parameters.T_max)
        
        return ChEESState(q=jnp.mean(positions, axis=0), mu=mu_new, T=T_new)
    
    def value(self, state: ChEESState) -> tuple[float, float]:
        """Get current integration time. Returns (unchanged_step_size, integration_time)."""
        return (self.passthrough_step_size, state.T)
    
    def finalize(self, state: ChEESState) -> tuple[float, float]:
        """Get final integration time. Returns (unchanged_step_size, integration_time)."""
        return (self.passthrough_step_size, state.T)

def welford_ema_mean(mu: jnp.ndarray, x: jnp.ndarray, momentum: float) -> jnp.ndarray:
   """Exponential moving average of mean across batch dimension 0.
   Args:
      mu: (..., d) previous mean
      x: (B, ..., d) batch of samples to incorporate
      momentum: in [0,1). new_mu = momentum*mu + (1-momentum)*mean(x)
   """
   batch_mean = jnp.mean(x, axis=0)
   return momentum * mu + (1.0 - momentum) * batch_mean

def chees_grad_T_euclidean(
    positions_initial: jnp.ndarray, 
    positions_proposed: jnp.ndarray, 
    momenta_proposed: jnp.ndarray, 
    integration_times: jnp.ndarray, 
    mu: jnp.ndarray, 
    accept: jnp.ndarray
) -> jnp.ndarray:
    """Per-iteration stochastic gradient estimator for d/dT of ChEES objective.
    Args:
        positions_initial: (B, d) start positions
        positions_proposed: (B, d) proposed end positions
        momenta_proposed: (B, d) end momenta (after final half step)
        integration_times:  (B,)  actual integration times used (u*T per chain)
        mu: (d,)   running mean used in criterion
        accept: (B,) accept indicators or probabilities
    Returns:
        scalar gradient estimate wrt T (averaged over batch)
    """
    dq0 = positions_initial - mu  # (B, d)
    dq1 = positions_proposed - mu  # (B, d)
    qnorm_diff = jnp.sum(dq1 * dq1, axis=-1) - jnp.sum(dq0 * dq0, axis=-1)  # (B,)
    inner = jnp.sum(dq1 * momenta_proposed, axis=-1)  # (B,)
    # Gradient estimator (matches whitened Euclidean formula): t * (Δ||·||^2) * <dq1, momenta_proposed>
    g_i = integration_times * qnorm_diff * inner  # (B,)
    g_i = g_i * accept  # damp by accept prob/indicator
    return jnp.mean(g_i)

def chees_grad_T_mahalanobis(
    positions_initial: jnp.ndarray, 
    positions_proposed: jnp.ndarray, 
    momenta_proposed: jnp.ndarray, 
    integration_times: jnp.ndarray, 
    mu: jnp.ndarray, 
    accept: jnp.ndarray
) -> jnp.ndarray:
    raise NotImplementedError()