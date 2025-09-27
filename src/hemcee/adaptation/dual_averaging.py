"""Dual-averaging utilities for adaptive step-size selection."""

from typing import NamedTuple
import jax.numpy as jnp
import jax.lax as lax

class DAState(NamedTuple):
    """Container for dual averaging state variables."""
    iteration: int
    step_size: jnp.ndarray
    H_bar: jnp.ndarray
    log_epsilon_bar: jnp.ndarray

class DAParameters(NamedTuple):
    """Container for dual averaging hyper-parameters."""
    target_accept: float
    t0: float
    mu: float
    gamma: float
    kappa: float

def da_cond_update(
    accept_prob: float,
    parameters: DAParameters,
    state: DAState,
) -> DAState:
    """Update the dual averaging state.

    Args:
        accept_prob (float): Acceptance probability of the current proposal.
        parameters (DAParameters): Dual averaging parameters.
        state (DAState): Previous dual averaging state.

    Returns:
        DAState: Updated dual averaging state.
    """
    it = state.iteration

    H_bar_new = ((1.0 - 1.0 / (it + 1 + parameters.t0)) * state.H_bar
                    + (parameters.target_accept - accept_prob) / (it + 1 + parameters.t0))
    log_eps = parameters.mu - (jnp.sqrt(it + 1.0) / parameters.gamma) * H_bar_new
    eta = (it + 1.0) ** (-parameters.kappa)
    log_eps_bar_new = eta * log_eps + (1.0 - eta) * state.log_epsilon_bar
    step_size_new = jnp.exp(log_eps)

    return DAState(it + 1, step_size_new, H_bar_new, log_eps_bar_new)

