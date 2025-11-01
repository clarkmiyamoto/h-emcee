import jax.numpy as jnp
from typing import NamedTuple

from hemcee.moves.hamiltonian.hmc_walk import leapfrog_walk_move

def u_turn_condition(r_current: jnp.ndarray, 
                          z_current: jnp.ndarray, 
                          z_initial: jnp.ndarray,
                          centering: jnp.ndarray,
                          inv_covariance: jnp.ndarray) -> jnp.ndarray:
    """
    Original No-U-Turn condition from Hoffman & Gelman 2014.
    
    U-turn detected if: r · (z_current - z_initial) ≤ 0
    
    Args:
        r_current: Current momentum. Shape (n_walkers_per_group,)
        z_current: Current position. Shape (dim,)
        z_initial: Initial position. Shape (dim,)
        centering: Centering matrix for momentum to position space conversion. Shape (dim, n_walkers_per_group)
        inv_covariance: Inverse covariance matrix for position space. Shape (dim, dim)
        
    Returns:
        Boolean indicating if U-turn is detected
    """
    # Calculate displacement vector
    displacement = z_current - z_initial
    
    # Calculate dot product: (z_current - z_initial)^T * inv_covariance * centering * r_current
    dot_product = displacement @ inv_covariance @ centering @ r_current
    
    # U-turn if dot product is negative or zero
    return dot_product <= 0