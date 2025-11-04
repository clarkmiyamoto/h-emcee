
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

def hmc_walk_move(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    grad_log_prob: Callable,
    L: int,
    log_prob_group1: jnp.ndarray,
):
    """Propose a Hamiltonian walk move.

    Implements Algorithm (3) from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complementary group with shape ``(n_chains_per_group, dim)``.
        step_size (float): Leapfrog step size.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorised log-probability function.
        grad_log_prob (Callable): Vectorised gradient of the potential function.
        L (int): Number of leapfrog steps.
        log_prob_group1 (jnp.ndarray): Log probabilities of the first group with shape ``(n_chains_per_group,)``.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and log acceptance
        probabilities for each chain.
    """
    n_chains_per_group = int(group1.shape[0])

    centered2 = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)
    momentum = jax.random.normal(key, shape=(n_chains_per_group, n_chains_per_group))
    
    # Leapfrog Integration
    group1_proposed, momentum_proposed, momentum_projected = leapfrog_walk_move(
        group1, 
        momentum, 
        grad_log_prob, 
        step_size, 
        L, 
        centered2
    )
    current_U = -1 * log_prob_group1 # Shape (n_chains_per_group,)
    current_K = 0.5 * jnp.sum(momentum**2, axis=1) # Shape (n_chains_per_group,)
    
    proposed_log_prob = log_prob(group1_proposed)
    proposed_U = -1 * proposed_log_prob
    proposed_K = 0.5 * jnp.sum(momentum_proposed**2, axis=1)

    dH = (proposed_U + proposed_K) - (current_U + current_K)
    dH = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    log_accept_prob1 = jnp.minimum(0.0, -dH)


    return group1_proposed, log_accept_prob1, momentum_projected, proposed_log_prob

def leapfrog_walk_move(
    q: jnp.ndarray,
    p: jnp.ndarray,
    grad_log_prob: Callable,
    beta_eps: float,
    L: int,
    centered: jnp.ndarray,
):
    """Perform leapfrog integration for the Hamiltonian walk move.

    Args:
        q (jnp.ndarray): Positions of the first group of chains with shape
            ``(n_chains_per_group, dim)``.
        p (jnp.ndarray): Momenta matrix with shape ``(n_chains_per_group, n_chains_per_group)``.
        grad_log_prob (Callable): Vectorised gradient of the log probability mapping
            ``(batch_size, dim)`` to ``(batch_size, dim)``.
        beta_eps (float): Step size scaled by ``beta``.
        L (int): Number of leapfrog steps.
        centered (jnp.ndarray): Centred complementary group with shape
            ``(n_chains_per_group, dim)``.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Updated positions and momenta.
    """
    grad = -1 * grad_log_prob(q) # Shape (n_chains_per_group, dim)
    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T) # Shape (n_chains_per_group, n_chains_per_group)
    
    # First L-1 steps with full momentum updates
    def leapfrog_step(step, state):
        q, p = state
        q += beta_eps * jnp.dot(p, centered) # Shape (n_chains_per_group, dim)
        grad = -1 * grad_log_prob(q) # Shape (n_chains_per_group, dim)
        p -= beta_eps * jnp.dot(grad, centered.T)
        return q, p
    
    q, p = jax.lax.fori_loop(0, L - 1, leapfrog_step, (q, p))
    
    # Last step: only update position, not momentum
    q += beta_eps * jnp.dot(p, centered)

    grad = -1 * grad_log_prob(q) # Shape (n_chains_per_group, dim)
    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

    # For logging purposes
    p_projected = p @ centered  # Shape (n_chains_per_group, dim)

    return q, p, p_projected