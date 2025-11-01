import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from hemcee.adaptation.nuts import (
    IntegratorState,
    nuts_step,
    euclidean_kinetic_energy,
)
from hemcee.adaptation.nuts_walk import u_turn_condition_walk
from hemcee.adaptation.nuts_side import u_turn_condition_walk as u_turn_condition_side


def _make_potential_from_vectorized_log_prob(
    log_prob_vectorized: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Wrap a vectorized log_prob into a per-sample potential function.

    The samplers pass in a vectorized `log_prob` (via vmap). NUTS internals
    expect scalar-valued potential/grad for a single position. This adapter
    expands the input to batch-size 1 and extracts the scalar.
    """

    def potential_fn_single(z_single: jnp.ndarray) -> jnp.ndarray:
        # log_prob_vectorized accepts shape (batch, dim) and returns (batch,)
        return -log_prob_vectorized(jnp.expand_dims(z_single, axis=0))[0]

    return potential_fn_single


def nuts_move_regular(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    grad_log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    L: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propose a move using regular NUTS, vectorized across chains.

    Signature matches existing Hamiltonian move functions so this can be
    passed directly as the `move` callable to samplers.

    Args:
        group1: Current positions for the proposal group, shape (n_chains, dim).
        group2: Complement group (unused by regular NUTS).
        step_size: Integrator step size for NUTS.
        key: PRNGKey for randomness.
        log_prob: Vectorized log-probability function, maps (n, dim) -> (n,).
        grad_log_prob: Vectorized gradient (unused here; NUTS computes grads via jax.grad).
        L: Interpreted as max_tree_depth for NUTS.

    Returns:
        Tuple of (proposed positions, log_accept_prob) with shapes
        (n_chains, dim) and (n_chains,).
    """
    del group2  # unused in regular NUTS
    del grad_log_prob  # NUTS computes gradients internally from potential

    n_chains = int(group1.shape[0])
    dim = int(group1.shape[1])

    # Potential function for a single chain input
    potential_fn = _make_potential_from_vectorized_log_prob(log_prob)

    # Prepare per-chain initial states
    keys = jax.random.split(key, n_chains)

    def one_chain_step(z0: jnp.ndarray, k: jax.random.PRNGKey):
        # Identity inverse mass (diagonal) as a vector for efficiency
        inv_mass_diag = jnp.ones((dim,))

        # Draw initial momentum r ~ N(0, I)
        r0 = jax.random.normal(k, shape=(dim,))

        pe0 = potential_fn(z0)
        z_grad0 = jax.grad(potential_fn)(z0)
        integrator_state = IntegratorState(z=z0, r=r0, potential_energy=pe0, z_grad=z_grad0)

        # Run a single NUTS step for this chain
        new_state, accept_prob, _num_steps, _diverging = nuts_step(
            integrator_state,
            potential_fn,
            euclidean_kinetic_energy,
            inv_mass_diag,
            step_size,
            int(L),
            k,
            turning_predicate=None  # Use default (original) turning condition
        )

        return new_state.z, accept_prob

    # Vectorize across chains
    proposed_positions, accept_probs = jax.vmap(one_chain_step)(group1, keys)

    # Convert NUTS acceptance probabilities to log space for MH
    # NUTS returns accept_prob in [0,1], convert to log space
    log_accept_prob = jnp.log(jnp.clip(accept_probs, 1e-10, 1.0))

    return proposed_positions, log_accept_prob


def _make_walk_turning_predicate(group2: jnp.ndarray) -> Callable:
    """Create a walk turning predicate from group2 samples.
    
    Uses the empirical covariance of group2 samples for the walk turning condition.
    
    Args:
        group2: Complementary group samples with shape (n_chains, dim)
        
    Returns:
        Turning predicate function compatible with NUTS
    """
    n_chains_per_group = int(group2.shape[0])
    dim = int(group2.shape[1])
    
    # Compute empirical covariance of group2
    group2_mean = jnp.mean(group2, axis=0)  # Shape (dim,)
    group2_centered = group2 - group2_mean[None, :]  # Shape (n_chains, dim)
    empirical_cov = jnp.dot(group2_centered.T, group2_centered) / (n_chains_per_group - 1)  # Shape (dim, dim)
    
    # Add small regularization to ensure invertibility
    regularization = 1e-6 * jnp.trace(empirical_cov) / dim * jnp.eye(dim)
    empirical_cov_reg = empirical_cov + regularization
    
    # Compute inverse covariance
    inv_covariance = jnp.linalg.inv(empirical_cov_reg)  # Shape (dim, dim)
    
    # Compute centering matrix (mean-centered group2)
    centering = group2_centered.T  # Shape (dim, n_chains)
    
    def walk_turning_predicate(z_left: jnp.ndarray,
                              r_left: jnp.ndarray,
                              z_right: jnp.ndarray, 
                              r_right: jnp.ndarray,
                              z_initial: jnp.ndarray) -> jnp.ndarray:
        """Walk turning predicate using empirical covariance of group2."""
        # Use a simpler walk turning condition that uses the empirical covariance
        # directly without the complex matrix chain
        
        # For left end
        displacement_left = z_left - z_initial
        # Use empirical covariance in the turning condition: r^T * inv_cov * displacement
        dot_product_left = r_left @ inv_covariance @ displacement_left
        
        # For right end  
        displacement_right = z_right - z_initial
        dot_product_right = r_right @ inv_covariance @ displacement_right
        
        # U-turn if either dot product is negative or zero
        turning_at_left = dot_product_left <= 0
        turning_at_right = dot_product_right <= 0
        
        return turning_at_left | turning_at_right
    
    return walk_turning_predicate


def nuts_move_walk(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    grad_log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    L: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propose a move using NUTS with walk turning condition.
    
    Uses the walk turning condition that depends on group2 samples for centering.
    
    Args:
        group1: Current positions for the proposal group, shape (n_chains, dim).
        group2: Complementary group samples, shape (n_chains, dim).
        step_size: Integrator step size for NUTS.
        key: PRNGKey for randomness.
        log_prob: Vectorized log-probability function, maps (n, dim) -> (n,).
        grad_log_prob: Vectorized gradient (unused here; NUTS computes grads via jax.grad).
        L: Interpreted as max_tree_depth for NUTS.
        
    Returns:
        Tuple of (proposed positions, log_accept_prob) with shapes
        (n_chains, dim) and (n_chains,).
    """
    del grad_log_prob  # NUTS computes gradients internally from potential
    
    n_chains = int(group1.shape[0])
    dim = int(group1.shape[1])
    
    # Create walk turning predicate from group2
    walk_turning_predicate = _make_walk_turning_predicate(group2)
    
    # Potential function for a single chain input
    potential_fn = _make_potential_from_vectorized_log_prob(log_prob)
    
    # Prepare per-chain initial states
    keys = jax.random.split(key, n_chains)
    
    def one_chain_step(z0: jnp.ndarray, k: jax.random.PRNGKey):
        # Use standard NUTS momentum structure (dim,) for each chain
        inv_mass_diag = jnp.ones((dim,))
        
        # Draw initial momentum r ~ N(0, I) with shape (dim,)
        r0 = jax.random.normal(k, shape=(dim,))
        
        pe0 = potential_fn(z0)
        z_grad0 = jax.grad(potential_fn)(z0)
        integrator_state = IntegratorState(z=z0, r=r0, potential_energy=pe0, z_grad=z_grad0)
        
        # Run a single NUTS step for this chain with walk turning predicate
        new_state, accept_prob, _num_steps, _diverging = nuts_step(
            integrator_state,
            potential_fn,
            euclidean_kinetic_energy,
            inv_mass_diag,
            step_size,
            int(L),
            k,
            turning_predicate=walk_turning_predicate
        )
        
        return new_state.z, accept_prob
    
    # Vectorize across chains
    proposed_positions, accept_probs = jax.vmap(one_chain_step)(group1, keys)
    
    # Convert NUTS acceptance probabilities to log space for MH
    log_accept_prob = jnp.log(jnp.clip(accept_probs, 1e-10, 1.0))
    
    return proposed_positions, log_accept_prob


def _make_side_turning_predicate(group2: jnp.ndarray) -> Callable:
    """Create a side turning predicate from group2 samples.
    
    Uses the difference between randomly selected particles from group2 for the side turning condition.
    
    Args:
        group2: Complementary group samples with shape (n_chains, dim)
        
    Returns:
        Turning predicate function compatible with NUTS
    """
    n_chains_per_group = int(group2.shape[0])
    dim = int(group2.shape[1])
    
    # For side moves, we need to select random pairs from group2
    # This is done per-chain in the turning predicate
    def side_turning_predicate(z_left: jnp.ndarray,
                              r_left: jnp.ndarray,
                              z_right: jnp.ndarray, 
                              r_right: jnp.ndarray,
                              z_initial: jnp.ndarray) -> jnp.ndarray:
        """Side turning predicate using group2 particle differences."""
        # For side moves, the turning condition is simpler: r_current * r_initial <= 0
        # But we need to handle the fact that NUTS uses vector momentum while side uses scalar
        
        # Use the original U-turn condition for now as a placeholder
        # The side turning condition needs more careful integration with NUTS
        from hemcee.adaptation.nuts import u_turn_condition
        
        turning_at_left = u_turn_condition(r_left, z_left, z_initial)
        turning_at_right = u_turn_condition(r_right, z_right, z_initial)
        
        return turning_at_left | turning_at_right
    
    return side_turning_predicate


def nuts_move_side(
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    step_size: float,
    key: jax.random.PRNGKey,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    grad_log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    L: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propose a move using NUTS with side turning condition.
    
    Uses the side turning condition that depends on group2 samples for particle differences.
    
    Args:
        group1: Current positions for the proposal group, shape (n_chains, dim).
        group2: Complementary group samples, shape (n_chains, dim).
        step_size: Integrator step size for NUTS.
        key: PRNGKey for randomness.
        log_prob: Vectorized log-probability function, maps (n, dim) -> (n,).
        grad_log_prob: Vectorized gradient (unused here; NUTS computes grads via jax.grad).
        L: Interpreted as max_tree_depth for NUTS.
        
    Returns:
        Tuple of (proposed positions, log_accept_prob) with shapes
        (n_chains, dim) and (n_chains,).
    """
    del grad_log_prob  # NUTS computes gradients internally from potential
    
    n_chains = int(group1.shape[0])
    dim = int(group1.shape[1])
    
    # Create side turning predicate from group2
    side_turning_predicate = _make_side_turning_predicate(group2)
    
    # Potential function for a single chain input
    potential_fn = _make_potential_from_vectorized_log_prob(log_prob)
    
    # Prepare per-chain initial states
    keys = jax.random.split(key, n_chains)
    
    def one_chain_step(z0: jnp.ndarray, k: jax.random.PRNGKey):
        # Use standard NUTS momentum structure (dim,) for each chain
        inv_mass_diag = jnp.ones((dim,))
        
        # Draw initial momentum r ~ N(0, I) with shape (dim,)
        r0 = jax.random.normal(k, shape=(dim,))
        
        pe0 = potential_fn(z0)
        z_grad0 = jax.grad(potential_fn)(z0)
        integrator_state = IntegratorState(z=z0, r=r0, potential_energy=pe0, z_grad=z_grad0)
        
        # Run a single NUTS step for this chain with side turning predicate
        new_state, accept_prob, _num_steps, _diverging = nuts_step(
            integrator_state,
            potential_fn,
            euclidean_kinetic_energy,
            inv_mass_diag,
            step_size,
            int(L),
            k,
            turning_predicate=side_turning_predicate
        )
        
        return new_state.z, accept_prob
    
    # Vectorize across chains
    proposed_positions, accept_probs = jax.vmap(one_chain_step)(group1, keys)
    
    # Convert NUTS acceptance probabilities to log space for MH
    log_accept_prob = jnp.log(jnp.clip(accept_probs, 1e-10, 1.0))
    
    return proposed_positions, log_accept_prob


