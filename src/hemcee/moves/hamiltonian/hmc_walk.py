
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

def hmc_walk_move(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable,
    grad_potential_func_vmap: Callable,
    L: int,
):
    """Propose a Hamiltonian walk move.

    Implements Algorithm (3) from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complementary group with shape ``(n_chains_per_group, dim)``.
        step_size (float): Leapfrog step size.
        key (jax.random.PRNGKey): Random number generator key.
        potential_func_vmap (Callable): Vectorised potential energy function.
        grad_potential_func_vmap (Callable): Vectorised gradient of the potential function.
        L (int): Number of leapfrog steps.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and log acceptance
        probabilities for each chain.
    """
    n_chains_per_group = int(group1.shape[0])

    key_momentum, key_accept = jax.random.split(key, 2)
    centered2 = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)
    momentum = jax.random.normal(key_momentum, shape=(n_chains_per_group, n_chains_per_group))
    
    # Leapfrog Integration
    group1_proposed, momentum_proposed = leapfrog_walk_move(
        group1, 
        momentum, 
        grad_potential_func_vmap, 
        step_size, 
        L, 
        centered2
    )
    current_U = potential_func_vmap(group1) # Shape (n_chains_per_group,)
    current_K = 0.5 * jnp.sum(momentum**2, axis=1) # Shape (n_chains_per_group,)
    
    proposed_U = potential_func_vmap(group1_proposed)
    proposed_K = 0.5 * jnp.sum(momentum_proposed**2, axis=1)

    dH = (proposed_U + proposed_K) - (current_U + current_K)
    log_accept_prob1 = jnp.minimum(0.0, -dH)

    return group1_proposed, log_accept_prob1

def leapfrog_walk_move(
    q: jnp.ndarray,
    p: jnp.ndarray,
    grad_fn: Callable,
    beta_eps: float,
    L: int,
    centered: jnp.ndarray,
):
    """Perform leapfrog integration for the Hamiltonian walk move.

    Args:
        q (jnp.ndarray): Positions of the first group of chains with shape
            ``(n_chains_per_group, dim)``.
        p (jnp.ndarray): Momenta matrix with shape ``(n_chains_per_group, n_chains_per_group)``.
        grad_fn (Callable): Vectorised gradient of the log probability mapping
            ``(batch_size, dim)`` to ``(batch_size, dim)``.
        beta_eps (float): Step size scaled by ``beta``.
        L (int): Number of leapfrog steps.
        centered (jnp.ndarray): Centred complementary group with shape
            ``(n_chains_per_group, dim)``.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Updated positions and momenta.
    """
    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0) 

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T) # Shape (n_chains_per_group, n_chains_per_group)
   

    for step in range(L):
        q += beta_eps * jnp.dot(p, centered) # Shape (n_chains_per_group, dim)

        if (step < L - 1):
            grad = grad_fn(q) # Shape (n_chains_per_group, dim)
            grad = jnp.nan_to_num(grad, nan=0.0)

            p -= beta_eps * jnp.dot(grad, centered.T)

    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0)

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

    return q, p
    
