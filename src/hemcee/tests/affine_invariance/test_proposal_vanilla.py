"""Tests for proposal functions and move implementations."""
import pytest

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hemcee.moves.vanilla.side import side_move
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.moves.vanilla.walk import walk_move

from hemcee.tests.affine_invariance.tools import make_distributions_and_affine_transformation

dim = 5
n_chains_per_group = dim * 4

@pytest.mark.parametrize(
    "move_fn",
    [
        side_move,
        walk_move,
        stretch_move,
    ],
)
def test_affineInvariance_VanillaMoves(move_fn):
    for seed in [0, 1, 2]:
        print(f"Running {move_fn.__name__} with seed {seed}")
        run(move_fn, seed)

def run(move_fn, seed):
    """
    Check proposal is affine invariant.
    Does check described in https://arxiv.org/pdf/2505.02987 (page 4).
    _____________________________________________________________

    Let R be your one-step proposal operator that takes ensemble X = (x_1, x_2, ..., x_N) and returns X' = R(X; pi).
    Let phi be the affine map phi(x) = A @ x + b.

    The proposal R is affine invariant when
    R(A . X + b; phi_#(pi)) = A . R(X; pi) + b
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    log_prob, log_prob_pushforward, A, b = make_distributions_and_affine_transformation(keys[2], dim)

    ### VMAP distributions
    log_prob_func_vmap = jax.vmap(log_prob)
    log_prob_pushforward_func_vmap = jax.vmap(log_prob_pushforward)

    ### Initialize walkers
    group1 = jax.random.normal(keys[4], shape=(n_chains_per_group, dim), dtype=jnp.float64)
    group2 = jax.random.normal(keys[5], shape=(n_chains_per_group, dim), dtype=jnp.float64)

    group1_transformed = jnp.einsum('ji,bi->bj', A, group1) + b[None, :] # Shape (n_chains_per_group, dim)
    group2_transformed = jnp.einsum('ji,bi->bj', A, group2) + b[None, :] # Shape (n_chains_per_group, dim)

    ### Proposal Paths
    # Path A: Original coordinates, then affine transformation
    group1_proposed_A, _ = move_fn(group1, group2, keys[6], log_prob_func_vmap)
    group1_proposed_A = jnp.einsum('ji,bi->bj', A, group1_proposed_A) + b[None, :]

    # Path B: Affine transformation, then original coordinates
    group1_proposed_B, _ = move_fn(group1_transformed, group2_transformed, keys[6], log_prob_pushforward_func_vmap)

    ### Check equality for affine invariant moves
    assert jnp.allclose(group1_proposed_A, group1_proposed_B)


