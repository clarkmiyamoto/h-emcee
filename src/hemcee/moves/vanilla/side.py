import jax
import jax.numpy as jnp
from typing import Callable

def side_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    log_prob_vmap: Callable,
    stretch: float = 2.0):
    '''
    Side Move sampler implementation using JAX.
    Algorithm (2) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Shape (n_chains_per_group, dim)
        group2: Shape (n_chains_per_group, dim)
        key: JAX random key
        log_prob_vmap: Log probability function vectorized
        stretch: Stretch parameter, must be >= 1. Default to 2.
    '''
    n_chains_per_group = int(group1.shape[0])

    keys = jax.random.split(key, n_chains_per_group + 2)
    key_choices = keys[0:n_chains_per_group]
    key_noise = keys[-2]
    key_accept = keys[-1]

    indices = jnp.arange(n_chains_per_group)
    choices = jax.vmap(
        lambda k: jax.random.choice(
            k, indices, shape=(2,), replace=False)
        )(key_choices) # Shape (n_chains_per_group, 2)
    
    random_indices1_from_group2 = choices[:, 0]
    random_indices2_from_group2 = choices[:, 1]

    z = jax.random.normal(key_noise, shape=(n_chains_per_group, 1))

    xj = group2[random_indices1_from_group2] # Shape (n_chains_per_group, dim)
    xk = group2[random_indices2_from_group2] # Shape (n_chains_per_group, dim)

    group1_proposed = group1 + stretch * z * (xj - xk)

    log_accept_prob = log_prob_vmap(group1) - log_prob_vmap(group1_proposed)
    log_u = jnp.log(jax.random.uniform(key_accept, shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
    accept_mask = log_u < log_accept_prob
    updated_group1_states = jnp.where(accept_mask[:, None], group1_proposed, group1)

    return updated_group1_states, accept_mask