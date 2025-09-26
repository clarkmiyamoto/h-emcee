import jax
import jax.numpy as jnp
from typing import Callable

def walk_move(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    **kwargs,
):
    """Propose a walk move for the affine-invariant sampler.

    Implements Algorithm (4) from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complementary group with shape ``(n_chains_per_group, dim)``.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorised log-probability function.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and log acceptance
        probabilities for each chain.
    """
    key_noise, key_accept = jax.random.split(key, 2)
    num_chains_per_group = int(group1.shape[0])

    m = jnp.mean(group2, axis=0)                     # (dim,)
    xi = jax.random.normal(key_noise, (num_chains_per_group,))    # (n_chains,)
    
    noisy_centered = jnp.einsum('bi, b -> i', group2 - m, xi) / jnp.sqrt(num_chains_per_group) # (dim,)
    group1_proposed = group1 + noisy_centered[None, :] # (n_chains, dim)

    log_accept_prob = log_prob(group1_proposed) - log_prob(group1)
    
    return group1_proposed, log_accept_prob
