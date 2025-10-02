import jax.numpy as jnp
from typing import NamedTuple

from hemcee.moves.hamiltonian.hmc_walk import leapfrog_walk_move

class State(NamedTuple):
    position: jnp.ndarray
    momentum: jnp.ndarray

# Leapfrog + U-Turn
def is_u_turn(state_left: State, state_right: State, complement_ensemble: jnp.ndarray) -> bool:
    """Return True if a U-turn is detected between z_left and z_right."""
    n_complement = complement_ensemble.shape[0]
    delta_theta = state_left.position - state_right.position
    
    # Convert ensemble momentum to position space
    complement_mean = jnp.mean(complement_ensemble, axis=0)
    centered_complement = (complement_ensemble - complement_mean) / jnp.sqrt(n_complement)
    precision = centered_complement @ centered_complement.T
    
    change_in_position_cov = jnp.linalg.solve(precision, delta_theta)
    
    # Weighted inner products: delta_theta^T * cov_inv * p
    p_plus = state_left.momentum @ centered_complement
    p_minus = state_right.momentum @ centered_complement

    dot_plus = change_in_position_cov @ p_plus
    dot_minus = change_in_position_cov @ p_minus
    
    return (dot_plus >= 0) & (dot_minus >= 0)

def leapfrog(state: State, grad_log_prob: Callable, step_size: float, centered_complement: jnp.ndarray) -> State:
    """One leapfrog (or any reversible symplectic) step from z."""
    return leapfrog_walk_move(state.position, state.momentum, grad_log_prob, step_size, 1, centered_complement)