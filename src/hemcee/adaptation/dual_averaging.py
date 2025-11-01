"""Dual-averaging utilities for adaptive step-size selection."""

from typing import NamedTuple

import jax.numpy as jnp
from .base import Adapter

class DAState(NamedTuple):
    """Container for dual averaging state variables."""
    iteration: int
    log_stepsize: float
    log_stepsize_bar: float
    H_bar: float


class DAParameters(NamedTuple):
    """
    Container for dual averaging hyper-parameters.
    
    Attributes:
        target_accept (float): Target acceptance probability.
        t0 (float): Free parameter that stabilizes initial iterations.
        mu (float): Log of the initial step size.
        gamma (float): Controls the speed of adaptation.
        kappa (float): Controls the shrinkage towards the average.
    """
    target_accept: float = 0.651
    stepsize_inter: float = 0.9
    t0: float = 10.0
    gamma: float = 0.05
    kappa: float = 0.75
    agg: str = 'harmonic' # 'mean' or 'harmonic'


class DualAveragingAdapter(Adapter):
    """Dual averaging adapter for step size adaptation."""
    
    def __init__(self, parameters: DAParameters, initial_step_size: float, initial_L: float):
        self.parameters = parameters
        self.initial_step_size = initial_step_size
        self.passthrough_L = initial_L
        
        if parameters.agg == 'mean':
            self.average_over_rate = lambda accept_log_prob: jnp.mean(
                jnp.clip(jnp.exp(accept_log_prob), min=0.0, max=1.0)
            )
        elif parameters.agg == 'harmonic':
            self.average_over_rate = lambda accept_log_prob: accept_log_prob.size / jnp.sum(
                1/ jnp.clip(jnp.exp(accept_log_prob), min=0.0, max=1.0)
            )
        else:
            raise ValueError('DAParmeter.agg must be either "mean" or "harmonic"')

    
    def init(self, dim: int) -> DAState:
        """Initialize dual averaging state."""
        return DAState(
            iteration = 0,
            log_stepsize = jnp.log(self.initial_step_size),
            log_stepsize_bar = jnp.log(self.initial_step_size),
            H_bar = 0.0,
        )

    def update(self, 
               state: DAState, 
               log_accept_rate: jnp.ndarray,
               **kwargs) -> DAState:
        """Update step size using dual averaging (positions ignored)."""
        it = state.iteration + 1

        # Get harmonic mean acceptance rate
        accept_rate = self.average_over_rate(log_accept_rate)
        
        # Dual averaging update
        eta = 1.0 / (it + self.parameters.t0)
        H_bar = state.H_bar + (self.parameters.target_accept - accept_rate)

        soft_t = it + self.parameters.t0
        log_eps = self.initial_step_size - (jnp.sqrt(it) / (soft_t * self.parameters.gamma)) * H_bar
        
        eta = it ** (-self.parameters.kappa)
        log_eps_bar = eta * log_eps + (1.0 - eta) * state.log_stepsize_bar

        return DAState(
            iteration=it, 
            log_stepsize=log_eps, 
            log_stepsize_bar=log_eps_bar, 
            H_bar=H_bar
        )

    def value(self, state: DAState) -> tuple[float, float]:
        """Get current step size during warmup. Returns (step_size, unchanged_integration_time)."""
        return (jnp.exp(state.log_stepsize), self.passthrough_L)  # Default integration time, unchanged
    
    def finalize(self, state: DAState) -> tuple[float, float]:
        """Get final step size from dual average. Returns (step_size, unchanged_integration_time)."""
        return (jnp.exp(state.log_stepsize_bar), self.passthrough_L)  # Default integration time, unchanged