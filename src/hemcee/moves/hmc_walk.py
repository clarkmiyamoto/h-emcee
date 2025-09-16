
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

def hmc_walk(
    group1: jnp.ndarray, group2: jnp.ndarray,
    da_state, 
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable, grad_potential_func_vmap: Callable,
    L: int):
    '''
    Hamiltonian Walk Move (HWM) sampler implementation using JAX.
    Algorithm (3) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Proposal group. Shape (n_chains_per_group, dim)
        group2: Complement group. Shape (n_chains_per_group, dim)
        da_state: Dual Averaging State
        key: JAX random key
        n_chains_per_group: Number of chains per group (total chains = 2 * n_chains_per_group).
        potential_func_vmap: Potential function vectorized
        grad_potential_func_vmap: Gradient of potential function vectorized
        L: Number of leapfrog steps
    '''
    n_chains_per_group = int(group1.shape[0])

    key_momentum, key_accept = jax.random.split(key, 2)
    centered2 = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)
    momentum = jax.random.normal(key_momentum, shape=(n_chains_per_group, n_chains_per_group))
    
    # Leapfrog Integration
    group1_proposed, momentum_proposed = leapfrog_walk_move(
        group1, 
        momentum, 
        grad_potential_func_vmap, 
        da_state.step_size, 
        L, 
        centered2
    )
    current_U = potential_func_vmap(group1) # Shape (n_chains_per_group,)
    current_K = 0.5 * jnp.sum(momentum**2, axis=1) # Shape (n_chains_per_group,)
    
    proposed_U = potential_func_vmap(group1_proposed)
    proposed_K = 0.5 * jnp.sum(momentum_proposed**2, axis=1)

    dH = (proposed_U + proposed_K) - (current_U + current_K)

    log_accept_prob1 = jnp.minimum(0.0, -dH)

    log_u1 = jnp.log(
        jax.random.uniform(
            key_accept, shape=(n_chains_per_group,), minval=1e-10, maxval=1.0
        )
    ) # shape (n_chains_per_group,)
    
    accepts = log_u1 < log_accept_prob1

    # Log Changes
    accepts = accepts.astype(int)
    group1 = jnp.where(accepts[:, None], group1_proposed, group1) # shape (n_chains_per_group, dim)

    return group1, accepts

def leapfrog_walk_move(q: jnp.ndarray, 
                       p: jnp.ndarray, 
                       grad_fn: Callable, 
                       beta_eps: float, 
                       L: int,
                       centered: jnp.ndarray):
    '''
    Args:
        q: Shape (n_chains_per_group, dim)
        p: Shape (n_chinas_per_group, n_chains_per_group)
        grad_fn: Gradient of log probabiltiy vectorized. Maps (batch_size, dim) -> (batch_size, dim)
        beta_eps: beta times step size (step_size)
        L: Number of steps
        centered: Shape (n_chains_per_group, dim)
    '''
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
    