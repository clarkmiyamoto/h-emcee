"""Dual-averaging utilities for adaptive step-size selection."""

from typing import NamedTuple
import jax.numpy as jnp
import jax.lax as lax

class DAState(NamedTuple):
    """Container for dual averaging state variables."""
    iteration: int
    step_size: int
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
    target_accept: float
    t0: float
    mu: float
    gamma: float
    kappa: float

def init_da_state(step_size: float) -> DAState:
    """Initialize the dual averaging state.

    Args:
        step_size (float): Initial step size.

    Returns:
        DAState: Initialized dual averaging state.
    """
    return DAState(
        iteration=0,
        step_size=step_size,
        H_bar=0.0,
        log_epsilon_bar=jnp.log(step_size),
    )

def init_da_parameters(step_size) -> DAParameters:
    """Initialize dual averaging parameters with default values.

    Args:
        step_size (float): Initial step size.

    Returns:
        DAParameters: Initialized dual averaging parameters.
    """
    return DAParameters(
        target_accept=0.8,
        t0=10.0,
        mu=jnp.log(10 * step_size),
        gamma=0.05,
        kappa=0.75,
    )

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

