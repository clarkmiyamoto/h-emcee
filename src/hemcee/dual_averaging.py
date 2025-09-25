"""Dual-averaging utilities for adaptive step-size selection."""

from typing import NamedTuple
import jax.numpy as jnp
import jax.lax as lax


class DAState(NamedTuple):
    """Container for dual averaging state variables."""

    step_size: jnp.ndarray
    H_bar: jnp.ndarray
    log_epsilon_bar: jnp.ndarray


def da_cond_update(
    iteration: int,
    warmup_length: int,
    accept_prob: float,
    target_accept: float,
    t0: float,
    mu: float,
    gamma: float,
    kappa: float,
    state: DAState,
) -> DAState:
    """Update the dual averaging state.

    Args:
        iteration (int): Current iteration index.
        warmup_length (int): Number of warmup iterations.
        accept_prob (float): Acceptance probability of the current proposal.
        target_accept (float): Target acceptance probability.
        t0 (float): Stability constant used in dual averaging.
        mu (float): Log step-size offset.
        gamma (float): Shrinkage parameter controlling adaptation speed.
        kappa (float): Exponent controlling the decay rate of adaptation.
        state (DAState): Previous dual averaging state.

    Returns:
        DAState: Updated dual averaging state.
    """

    def in_warmup(current: DAState) -> DAState:
        it = iteration
        H_bar_new = ((1.0 - 1.0 / (it + 1 + t0)) * current.H_bar
                     + (target_accept - accept_prob) / (it + 1 + t0))
        log_eps = mu - (jnp.sqrt(it + 1.0) / gamma) * H_bar_new
        eta = (it + 1.0) ** (-kappa)
        log_eps_bar_new = eta * log_eps + (1.0 - eta) * current.log_epsilon_bar
        step_size_new = jnp.exp(log_eps)
        return DAState(step_size_new, H_bar_new, log_eps_bar_new)

    def after_warmup(current: DAState) -> DAState:
        return DAState(jnp.exp(current.log_epsilon_bar), current.H_bar, current.log_epsilon_bar)

    return lax.cond(iteration < warmup_length, in_warmup, after_warmup, state)