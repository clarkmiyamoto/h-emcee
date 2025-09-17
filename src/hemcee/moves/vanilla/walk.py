import jax
import jax.numpy as jnp
from typing import Callable

def walk_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable,
    **kwargs):
    key_noise, key_accept = jax.random.split(key, 2)
    num_chains_per_group = int(group1.shape[0])

    m = jnp.mean(group2, axis=0)                     # (dim,)
    xi = jax.random.normal(key_noise, (num_chains_per_group,))    # (n_chains,)
    centered = group2 - m                             # (n_chains, dim)
    group1_proposed = group1 + jnp.sum(centered * xi[:, None], axis=0) / jnp.sqrt(num_chains_per_group)

    log_accept_prob = potential_func_vmap(group1) - potential_func_vmap(group1_proposed)
    log_u = jnp.log(jax.random.uniform(key_accept, shape=(num_chains_per_group,), minval=1e-10, maxval=1.0))
    accept_mask = log_u < log_accept_prob
    updated_group1_states = jnp.where(accept_mask[:, None], group1_proposed, group1)

    return updated_group1_states, accept_mask