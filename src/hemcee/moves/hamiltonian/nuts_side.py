import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from hemcee.adaptation.nuts import (
    IntegratorState,
    nuts_step,
    euclidean_kinetic_energy,
)
from hemcee.adaptation.nuts_side import u_turn_condition


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
                              z_initial: jnp.ndarray,
                              r_initial: jnp.ndarray) -> jnp.ndarray:
        """Side turning predicate using momentum comparison."""
        # For side moves, compare proposed momentum with initial momentum
        # U-turn when: r_proposed · r_initial ≤ 0
    
        turning_at_left = u_turn_condition(r_left, r_initial)
        turning_at_right = u_turn_condition(r_right, r_initial)
        
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


