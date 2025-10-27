"""Dual-averaging utilities for adaptive step-size selection."""

from typing import NamedTuple

import jax.numpy as jnp
from .base import Adapter

class DAState(NamedTuple):
    """Container for dual averaging state variables."""
    iteration: int
    step_size: float
    H_bar: float
    log_epsilon_bar: float

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
    target_accept: float = 0.8
    t0: float = 10.0
    gamma: float = 0.05
    kappa: float = 0.75


class DualAveragingAdapter(Adapter):
    """Dual averaging adapter for step size adaptation."""
    
    def __init__(self, parameters: DAParameters, initial_step_size: float, initial_L: float):
        self.parameters = parameters
        self.initial_step_size = initial_step_size
        self.passthrough_L = initial_L
    
    def init(self, dim: int) -> DAState:
        """Initialize dual averaging state."""
        return DAState(
            iteration=0,
            step_size=self.initial_step_size,
            H_bar=0.0,
            log_epsilon_bar=jnp.log(self.initial_step_size),
        )
    
    def update(self, state: DAState, accept_rate: float, positions: jnp.ndarray) -> DAState:
        """Update step size using dual averaging (positions ignored)."""
        it = state.iteration
        
        # Dual averaging update
        H_bar_new = (
            (1.0 - 1.0 / (it + 1 + self.parameters.t0)) * state.H_bar
            + (self.parameters.target_accept - accept_rate) / (it + 1 + self.parameters.t0)
        )
        log_eps = self.initial_step_size - (jnp.sqrt(it + 1.0) / self.parameters.gamma) * H_bar_new
        eta = (it + 1.0) ** (-self.parameters.kappa)
        log_eps_bar_new = eta * log_eps + (1.0 - eta) * state.log_epsilon_bar
        step_size_new = jnp.exp(log_eps)
        
        return DAState(it + 1, step_size_new, H_bar_new, log_eps_bar_new)
    
    def value(self, state: DAState) -> tuple[float, float]:
        """Get current step size during warmup. Returns (step_size, unchanged_integration_time)."""
        return (state.step_size, self.passthrough_L)  # Default integration time, unchanged
    
    def finalize(self, state: DAState) -> tuple[float, float]:
        """Get final step size from dual average. Returns (step_size, unchanged_integration_time)."""
        return (jnp.exp(state.log_epsilon_bar), self.passthrough_L)  # Default integration time, unchanged