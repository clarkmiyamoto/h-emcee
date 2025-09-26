import jax
import jax.numpy as jnp
from hemcee.tests.distribution import make_gaussian

def make_distributions_and_affine_transformation(key: jax.random.PRNGKey, dim: int):
    '''
    Make distributions and affine transformation.
    '''
    keys = jax.random.split(key, 2)

    # Affine transformation
    A = jax.random.normal(keys[0], shape=(dim, dim), dtype=jnp.float64)
    b = jax.random.normal(keys[1], shape=(dim,), dtype=jnp.float64)

    # Regular distribution
    log_prob = make_gaussian(jnp.eye(dim), jnp.zeros(dim))

    # Push-forward of target distribution
    info_pushforward = A @ A.T
    log_prob_pushforward = make_gaussian(info_pushforward, b)

    return log_prob, log_prob_pushforward, A, b