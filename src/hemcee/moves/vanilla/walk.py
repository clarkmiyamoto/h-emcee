import jax
import jax.numpy as jnp
from typing import Callable

def walk_move(
    group1: jnp.ndarray, group2: jnp.ndarray,
    key: jax.random.PRNGKey,
    potential_func_vmap: Callable,
    **kwargs):
    pass
    