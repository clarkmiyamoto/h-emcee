import jax
import jax.numpy as jnp
from typing import Callable

def stretch_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    log_prob_vmap: Callable,
    stretch: float = 2.0):
    '''
    Stretch Move sampler implementation using JAX.
    Algorithm (1) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Shape (n_chains_per_group, dim)
        group2: Shape (n_chains_per_group, dim)
        key: JAX random key
        log_prob_vmap: Log probability function vectorized
        stretch: Stretch parameter, must be >= 1. Default to 2.
    '''
    n_chains_per_group = int(group1.shape[0])
    dim = int(group1.shape[1])

    keys = jax.random.split(key, n_chains_per_group + 2)
    key_choices = keys[0:n_chains_per_group]
    key_stretch = keys[-1]
    key_accept = keys[-2]

    indices = jnp.arange(n_chains_per_group)
    choices = jax.vmap(
        lambda k: jax.random.choice(
            k, indices, shape=(1,), replace=False)
        )(key_choices) # Shape (n_chains_per_group, 1)
    
    random_indices = choices[:, 0]

    z = sample_inv_sqrt_density(key_stretch, a=stretch, shape=(n_chains_per_group,))
    group1_proposed = group2[random_indices] + z[:, None] * (group1 - group2[random_indices])

    log_accept_prob = (dim - 1) * jnp.log(z) + log_prob_vmap(group1) - log_prob_vmap(group1_proposed)

    return group1_proposed, log_accept_prob

def sample_inv_sqrt_density(key: jax.random.PRNGKey, a: float, shape=()):
    """
    Sample from density proportional to z^{-1/2} on [1/a, a], with a >= 1.
    """
    a = jnp.asarray(a, dtype=jnp.float_)
    u = jax.random.uniform(key, shape=shape, dtype=jnp.float_)  # U(0,1)
    sqrt_a = jnp.sqrt(a)
    inv_sqrt_a = 1.0 / sqrt_a
    z = (inv_sqrt_a + u * (sqrt_a - inv_sqrt_a)) ** 2
    return z