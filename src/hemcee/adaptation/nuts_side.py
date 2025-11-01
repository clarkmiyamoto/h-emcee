import jax.numpy as jnp
from typing import NamedTuple

from hemcee.moves.hamiltonian.hmc_side import leapfrog_side_move

def u_turn_condition(r_current: jnp.ndarray, 
                     r_initial: jnp.ndarray) -> jnp.ndarray:
    """
    Side No-U-Turn condition for momentum comparison.
    
    U-turn detected if: r_current · r_initial ≤ 0
    
    Args:
        r_current: Current momentum vector
        r_initial: Initial momentum vector
        
    Returns:
        Boolean indicating if U-turn is detected
    """
    # Calculate dot product: r_current · r_initial
    dot_product = jnp.dot(r_current, r_initial)
    
    # U-turn if dot product is negative or zero
    return dot_product <= 0