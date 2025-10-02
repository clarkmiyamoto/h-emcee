import jax.numpy as jnp
from typing import NamedTuple

from hemcee.moves.hamiltonian.hmc_side import leapfrog_side_move

class State(NamedTuple):
    position: jnp.ndarray
    momentum: jnp.ndarray

# Leapfrog + U-Turn
def is_u_turn(z_left: State, z_right: State) -> bool:
    """Return True if a U-turn is detected between z_left and z_right."""
    momentum_left = z_left.momentum
    momentum_right = z_right.momentum
    
    return (momentum_left * momentum_right < 0)

def leapfrog(z: State, grad_log_prob: Callable, step_size: float, complement: jnp.ndarray) -> State:
    """One leapfrog (or any reversible symplectic) step from z."""
    return leapfrog_side_move(z.position, z.momentum, grad_log_prob, step_size, 1, complement)