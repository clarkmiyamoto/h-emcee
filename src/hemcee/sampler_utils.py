"""Utility functions for MCMC samplers."""

from typing import Optional, Callable, Any
import jax
import jax.numpy as jnp
from tqdm import tqdm

from hemcee.backend.backend import Backend

def calculate_batch_size(total_chains: int, dim: int, total_iterations: int, user_batch_size: Optional[int] = None) -> int:
    """Calculate batch size based on problem dimensions or user input.
    
    Args:
        total_chains (int): Total number of chains in the ensemble.
        dim (int): Dimensionality of the target distribution.
        total_iterations (int): Total number of iterations to run.
        user_batch_size (int, optional): User-specified batch size. If None, uses heuristic.
        
    Returns:
        int: Batch size to use for scanning.
    """
    if user_batch_size is not None:
        return user_batch_size
    # Heuristic: larger problems need smaller batches
    heuristic_size = max(50, 10000 // (total_chains * dim))
    return min(heuristic_size, total_iterations)


def batched_scan(body_fn: Callable, 
                 init_carry: Any, 
                 xs: jnp.ndarray, 
                 batch_size: int,
                 backend: Backend,
                 show_progress = False) -> tuple[Any, Any]:
    """Run jax.lax.scan in batches to reduce memory usage.
    
    Args:
        body_fn: Function to apply to each batch.
        init_carry: Initial carry state.
        xs: Input array to scan over.
        batch_size: Size of each batch.
        backend: Backend for storing intermediate results.
        show_progress: Whether to display a progress bar.
        
    Returns:
        Tuple[Any, Any]: Final carry state and backend.
    """
    # Determine number of batches
    num_iterations = xs.shape[0]
    num_batches = (num_iterations + batch_size - 1) // batch_size
    
    # Store outputs from each batch
    carry = init_carry

    # Show progress bar? 
    if show_progress:
        batch_iter = tqdm(range(num_batches))
    else:
        batch_iter = range(num_batches)

    # Process each batch
    for batch_idx in batch_iter:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_iterations)
        xs_batch = xs[start_idx:end_idx]
        
        carry, outputs = jax.lax.scan(body_fn, carry, xs_batch)
        
        # Extract coords, log_prob, and accepted from outputs
        coords, log_prob, accepted = outputs
        
        backend.save_slice(
            coords=coords,
            log_prob=log_prob,
            accepted=jnp.sum(accepted, axis=0),
            index=start_idx
        )
    
    # Concatenate all samples
    
    return carry


def accept_proposal(
    current_samples: jnp.ndarray,
    proposed_samples: jnp.ndarray,
    log_accept_prob: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
