import jax
import jax.numpy as jnp
from typing import Callable

'''
TODO: If there are nan's you want the program to crash
      Code must also be -inf safe.
'''

def hmc_side_move(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    log_prob: Callable,
    grad_log_prob: Callable,
    L: int,
):
    """Propose a Hamiltonian side move.

    Implements Algorithm (4) from https://arxiv.org/pdf/2505.02987.

    Args:
        group1 (jnp.ndarray): Proposal group with shape ``(n_chains_per_group, dim)``.
        group2 (jnp.ndarray): Complementary group with shape ``(n_chains_per_group, dim)``.
        step_size (float): Leapfrog step size.
        key (jax.random.PRNGKey): Random number generator key.
        log_prob (Callable): Vectorised log-probability function.
        grad_log_prob (Callable): Vectorised gradient of the log-probability function.
        L (int): Number of leapfrog steps.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and log acceptance
        probabilities for each chain.
    """
    n_chains_per_group = int(group1.shape[0])
    dim = int(group1.shape[1])

    keys = jax.random.split(key, n_chains_per_group + 2)
    key_choices = keys[0:n_chains_per_group]
    key_momentum = keys[-2]
    key_accept = keys[-1]


    indices = jnp.arange(n_chains_per_group)
    choices = jax.vmap(
        lambda k: jax.random.choice(
            k, indices, shape=(2,), replace=False)
        )(key_choices) # Shape (n_chains_per_group, 2)
    
    random_indices1_from_group2 = choices[:, 0]
    random_indices2_from_group2 = choices[:, 1]

    diff_particles_group2 = (group2[random_indices1_from_group2] - group2[random_indices2_from_group2]) / jnp.sqrt(2*dim) # Shape (n_chains_per_group, dim)

    momentum = jax.random.normal(key_momentum, shape=(n_chains_per_group,))

    group1_proposed, momentum_proposed = leapfrog_side_move(
            group1, 
            momentum, 
            grad_log_prob, 
            step_size, 
            L, 
            diff_particles_group2
    )

    current_U1 = -1 * log_prob(group1) # Shape (n_chains_per_group, dim) -> (n_chains_per_group,)
    current_K1 = 0.5 * momentum**2

    proposed_U1 = -1 * log_prob(group1_proposed) # Shape (n_chains_per_group,)
    proposed_K1 = 0.5 * momentum_proposed**2

    dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1) # Shape (n_chains_per_group,)
    log_accept_prob1 = jnp.minimum(0.0, -dH1)
        
    # Track acceptance for first chains
    return group1_proposed, log_accept_prob1


def leapfrog_side_move(
    q: jnp.ndarray,
    p: jnp.ndarray,
    grad_log_prob: Callable,
    beta_eps: float,
    L: int,
    diff_particles_group2: jnp.ndarray,
):
    """Perform leapfrog integration for the Hamiltonian side move.

    Args:
        q (jnp.ndarray): Positions of the first group of chains with shape
            ``(n_chains_per_group, dim)``.
        p (jnp.ndarray): Momenta of the first group of chains with shape
            ``(n_chains_per_group,)``.
        grad_log_prob (Callable): Vectorised gradient of the log-probability function
            mapping ``(n_chains, dim)`` to ``(n_chains, dim)``.
        beta_eps (float): Step size scaled by ``beta``.
        L (int): Number of leapfrog steps.
        diff_particles_group2 (jnp.ndarray): Differences between particles in
            the second group with shape ``(n_chains_per_group, dim)``.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Updated positions and momenta for the
        first group of chains.
    """
    # Initial half-step for momentum - VECTORIZED
    grad = -1 * grad_log_prob(q) # Shape (n_chains_per_group, dim)
    gradient_projections = jnp.sum(grad * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
    
    p -= 0.5 * beta_eps * gradient_projections # Shape (n_chains_per_group,)

    # Full leapfrog steps
    def leapfrog_step(step, state):
        q, p = state
        q += beta_eps * (jnp.expand_dims(p, axis=1) * diff_particles_group2) # Shape: (n_chains_per_group, dim)
        
        def update_momentum(p):
            grad = -1 * grad_log_prob(q) # Shape (n_chains_per_group, dim)
            gradient_projections = jnp.sum(grad * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
            return p - beta_eps * gradient_projections # Shape (n_chains_per_group,)

        p = jax.lax.cond(step < L - 1, update_momentum, lambda p: p, p)
        return q, p
    
    q, p = jax.lax.fori_loop(0, L, leapfrog_step, (q, p))
    
    # Final half-step for momentum - VECTORIZED 
    grad = -1 * grad_log_prob(q)
    gradient_projections = jnp.sum(grad * diff_particles_group2, axis=1)
    
    p -= 0.5 * beta_eps * gradient_projections

    return q, p # Shape (n_chains_per_group, dim), (n_chains_per_group,)
