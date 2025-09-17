import jax
import jax.numpy as jnp
from typing import Tuple

def accept_proposal(
    current_samples: jnp.ndarray,
    proposed_samples: jnp.ndarray, 
    log_accept_prob: jnp.ndarray, 
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    '''
    Accept or reject a proposal in metropolis-hastings step.
    Log space is used for numerical stability.
    
    Args:
        current_samples: Shape (n_chains, dim)
        proposed_samples: Shape (n_chains, dim)
        log_accept_prob: Shape (n_chains,)
        key: JAX random key

    Returns:
        updated_samples: Shape (n_chains, dim)
        accepts: Shape (n_chains,)
    '''
    log_u = jnp.log(jax.random.uniform(key, shape=log_accept_prob.shape, minval=1e-10, maxval=1.0))
    accept_mask = log_u < log_accept_prob
    updated_samples = jnp.where(accept_mask[:, None], proposed_samples, current_samples)
    
    accepts = accept_mask.astype(int)

    return updated_samples, accepts
