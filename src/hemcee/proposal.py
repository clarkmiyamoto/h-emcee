"""Metropolis-Hastings proposal utilities."""

import jax
import jax.numpy as jnp
from typing import Tuple

def accept_proposal(
    current_samples: jnp.ndarray,
    proposed_samples: jnp.ndarray,
    log_accept_prob: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Accept or reject proposals in a Metropolis-Hastings step.

    Log space is used for numerical stability.

    Args:
        current_samples (jnp.ndarray): Current ensemble state with shape
            ``(n_chains, dim)``.
        proposed_samples (jnp.ndarray): Proposed ensemble state with shape
            ``(n_chains, dim)``.
        log_accept_prob (jnp.ndarray): Log acceptance probabilities with shape
            ``(n_chains,)``.
        key (jax.random.PRNGKey): Random number generator key.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Accepted samples and acceptance
        indicators with shapes ``(n_chains, dim)`` and ``(n_chains,)``.
    """
    log_u = jnp.log(jax.random.uniform(key, shape=log_accept_prob.shape, minval=1e-10, maxval=1.0))
    accept_mask = log_u < log_accept_prob
    updated_samples = jnp.where(accept_mask[:, None], proposed_samples, current_samples)
    
    accepts = accept_mask.astype(int)

    return updated_samples, accepts
