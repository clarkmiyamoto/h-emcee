import jax
import jax.numpy as jnp
from typing import Callable

def stretch_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    stretch: float = 2.0):
    """Perform the affine-invariant stretch move.

    Implements Algorithm 1 from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Ensemble group being updated with shape
            ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complement ensemble group with the same shape as
            ``group1``.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorized log-probability function.
        stretch (float): Stretch parameter ``a`` satisfying ``a >= 1``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Proposed positions for ``group1`` and
            their log acceptance probabilities.
    """
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

    log_accept_prob = (dim - 1) * jnp.log(z) + log_prob(group1_proposed) - log_prob(group1) 

    return group1_proposed, log_accept_prob

def sample_inv_sqrt_density(key: jax.random.PRNGKey, a: float, shape=()):
    """Sample from the ``z^{-1/2}`` density on ``[1/a, a]``.

    Args:
        key (jax.random.PRNGKey): Random number generator key.
        a (float): Stretch parameter satisfying ``a >= 1``.
        shape (tuple, optional): Output shape. Defaults to ``()``.

    Returns:
        jnp.ndarray: Samples drawn from the inverse square-root density with
            the requested shape.
    """
    a = jnp.asarray(a, dtype=jnp.float_)
    u = jax.random.uniform(key, shape=shape, dtype=jnp.float_)  # U(0,1)
    sqrt_a = jnp.sqrt(a)
    inv_sqrt_a = 1.0 / sqrt_a
    z = (inv_sqrt_a + u * (sqrt_a - inv_sqrt_a)) ** 2
    return z