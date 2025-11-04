import jax
import jax.numpy as jnp
from typing import Callable

def side_move(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    log_prob_group1: jnp.ndarray,
    stretch: float = 2.0,
):
    """Propose a side move using the affine-invariant sampler.

    Implements Algorithm (2) from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complementary group with shape ``(n_chains_per_group, dim)``.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorised log-probability function.
        log_prob_group1 (jnp.ndarray): Log probabilities of the first group with shape ``(n_chains_per_group,)``.
        stretch (float): Stretch parameter; must be greater than or equal to ``1``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Proposed positions, log acceptance
        probabilities, and proposed log probabilities for each chain.
    """
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

    log_prob_group1_proposed = log_prob(group1_proposed)
    log_accept_prob = log_prob_group1_proposed - log_prob_group1
    
    return group1_proposed, log_accept_prob, log_prob_group1_proposed
