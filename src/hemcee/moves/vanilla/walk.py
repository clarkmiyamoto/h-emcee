import jax
import jax.numpy as jnp
from typing import Callable

def walk_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable,
    **kwargs):
    '''
    Walk Move sampler implementation using JAX.
    Algorithm (4) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Shape (n_chains_per_group, dim)
        group2: Shape (n_chains_per_group, dim)
        key: JAX random key
        potential_func_vmap: Potential function vectorized
    '''
    key_noise, key_accept = jax.random.split(key, 2)
    num_chains_per_group = int(group1.shape[0])

    m = jnp.mean(group2, axis=0)                     # (dim,)
    xi = jax.random.normal(key_noise, (num_chains_per_group,))    # (n_chains,)
    
    noisy_centered = jnp.einsum('bi, b -> i', group2 - m, xi) / jnp.sqrt(num_chains_per_group) # (dim,)
    group1_proposed = group1 + noisy_centered[None, :] # (n_chains, dim)

    log_accept_prob = potential_func_vmap(group1) - potential_func_vmap(group1_proposed)
    
    return group1_proposed, log_accept_prob