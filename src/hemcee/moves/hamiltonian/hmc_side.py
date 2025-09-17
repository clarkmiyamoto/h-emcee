import jax
import jax.numpy as jnp
from typing import Callable

def hmc_side_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    da_state, 
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable, grad_potential_func_vmap: Callable,
    L: int):
    '''
    Hamiltonian Side Move (HSM) sampler implementation using JAX.
    Algorithm (4) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Shape (n_chains_per_group, dim)
        group2: Shape (n_chains_per_group, dim)
        da_state: Dual Averaging State
        key: JAX random key
        potential_func_vmap: Potential function vectorized
        grad_potential_func_vmap: Gradient of potential function vectorized
        L: Number of leapfrog steps
    '''
    n_chains_per_group = int(group1.shape[0])

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

    diff_particles_group2 = (group2[random_indices1_from_group2] - group2[random_indices2_from_group2]) / jnp.sqrt(2*n_chains_per_group)

    momentum = jax.random.normal(key_momentum, shape=(n_chains_per_group,))

    group1_proposed, momentum_proposed = leapfrog_side_move(
            group1, 
            momentum, 
            grad_potential_func_vmap, 
            da_state.step_size, 
            L, 
            n_chains_per_group,
            diff_particles_group2
    )

    current_U1 = potential_func_vmap(group1) # Shape (n_chains_per_group, dim) -> (n_chains_per_group,)
    current_K1 = 0.5 * momentum**2

    proposed_U1 = potential_func_vmap(group1_proposed) # Shape (n_chains_per_group,)
    proposed_K1 = 0.5 * momentum_proposed**2

    dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1) # Shape (n_chains_per_group,)
    log_accept_prob1 = jnp.minimum(0.0, -dH1)

    log_u1 = jnp.log(jax.random.uniform(key_accept, shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
    accepts = log_u1 < log_accept_prob1 # Shape (n_chains_per_group,)

    updated_group1_states = jnp.where(accepts[:, None], group1_proposed, group1)
        
    # Track acceptance for first chains
    return updated_group1_states, accepts


def leapfrog_side_move(q1, 
                       p1_current, 
                       grad_fn, 
                       beta_eps, 
                       L,
                       n_chains_per_group,
                       diff_particles_group2):
    '''
    Args
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1: momentum of first group of chains. Shape (n_chains_per_group,)
        - grad_fn: gradient of the potential function. Function of shape (n_chains, dim) -> (n_chains, dim)
        - beta_eps: half of the step size. Scalar
        - L: number of leapfrog steps. Scalar
        - L: number of leapfrog steps
    
    Returns
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1_current: momentum of first group of chains. Shape (n_chains_per_group,)
    '''
    # Initial half-step for momentum - VECTORIZED
    grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)
    
    # Compute dot products between gradients and difference particles - VECTORIZED
    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
    p1_current -= 0.5 * beta_eps * gradient_projections # Shape (n_chains_per_group,)

    # Full leapfrog steps
    for step in range(L):
        q1 += beta_eps * (jnp.expand_dims(p1_current, axis=1) * diff_particles_group2) # Shape: (n_chains_per_group, dim)
        
        if (step < L - 1):
            grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
            grad1 = jnp.nan_to_num(grad1, nan=0.0)

            gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
            p1_current -= beta_eps * gradient_projections # Shape (n_chains_per_group,)
    
    # Final half-step for momentum - VECTORIZED 
    grad1 = grad_fn(q1)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)

    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
    p1_current -= 0.5 * beta_eps * gradient_projections

    return q1, p1_current # Shape (n_chains_per_group, dim), (n_chains_per_group,)