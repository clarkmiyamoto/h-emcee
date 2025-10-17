import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

def hmc(
    group1: jnp.ndarray,
    step_size: float,
    inv_mass_matrix: jnp.ndarray,
    L: int,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    grad_log_prob: Callable,
):
    """Regular Hamiltonian Monte Carlo move.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        step_size (float): Leapfrog step size.
        inv_mass_matrix (jnp.ndarray): Inverted mass matrix.
        L (int): Number of leapfrog steps.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorised log-probability function.
        grad_potential_func_vmap (Callable): Vectorised gradient of the potential function.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and log acceptance
        probabilities for each chain.
    """
    n_chains_per_group = int(group1.shape[0])

    key_momentum, key_accept = jax.random.split(key, 2)
    momentum = sample_mvn_from_precision(key_momentum, inv_mass_matrix, n=n_chains_per_group)  # Shape (n_chains_per_group, dim)
    
    # Leapfrog Integration
    group1_proposed, momentum_proposed = leapfrog(
        group1, 
        momentum, 
        step_size, 
        L,
        inv_mass_matrix,
        grad_log_prob, 
    )

    current_U = -1 * log_prob(group1) # Shape (n_chains_per_group,)
    current_K = 0.5 * jnp.sum(momentum * (momentum @ inv_mass_matrix), axis=1) # Shape (n_chains_per_group,)

    proposed_U = -1 * log_prob(group1_proposed) # Shape (n_chains_per_group,)
    proposed_K = 0.5 * jnp.sum(momentum_proposed * (momentum_proposed @ inv_mass_matrix), axis=1) # Shape (n_chains_per_group,)

    dH = (proposed_U + proposed_K) - (current_U + current_K)
    log_accept_prob1 = jnp.minimum(0.0, -dH)

    return group1_proposed, log_accept_prob1

def leapfrog(position, momentum, step_size, L, inv_mass_matrix, grad_log_prob):
    """Leapfrog integrator for Hamiltonian dynamics.
    
    Args:
        position (jnp.ndarray): Current positions, shape (n_chains, dim).
        momentum (jnp.ndarray): Current momenta, shape (n_chains, dim).
        step_size (float): Leapfrog step size.
        L (int): Number of leapfrog steps.
        inv_mass_matrix (jnp.ndarray): Inverted mass matrix. Shape (dim, dim).
        grad_log_prob (Callable): Vectorised gradient of the log probability mapping
    """
    q = position
    p = momentum

    p -= 0.5 * step_size * grad_log_prob(q)
    for _ in range(L):
        q = q + step_size * (p @ inv_mass_matrix)
        if _ != L - 1:
            p = p - step_size * grad_log_prob(q)
    p = p - 0.5 * step_size * grad_log_prob(q)

    return q, p

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

def sample_mvn_from_precision(key, precision, n=1, jitter=1e-8):
    """
    Sample n draws from N(mean, Σ) given precision Λ = Σ^{-1}.
    
    Args:
        key: jax.random.PRNGKey
        precision: (d,d) SPD precision matrix Λ
        n: number of samples to draw
        jitter: small diagonal added for numerical stability

    Returns:
        samples: (n,d) array
    """
    d = precision.shape[-1]
    # Symmetrize & stabilize a bit
    precision = 0.5 * (precision + precision.T) + jitter * jnp.eye(d, dtype=precision.dtype)
    # Cholesky of precision
    L = jnp.linalg.cholesky(precision)          # L @ L.T = Λ
    # Standard normals
    z = jax.random.normal(key, (n, d), dtype=precision.dtype)
    # Solve L^T x^T = z^T  -> x = L^{-T} z
    x = solve_triangular(L, z.T, lower=True, trans='T').T
    return x