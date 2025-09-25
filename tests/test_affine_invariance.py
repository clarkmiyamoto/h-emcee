"""Tests for proposal functions and move implementations."""
import pytest

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular
jax.config.update("jax_enable_x64", True)

from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.vanilla.side import side_move
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.moves.vanilla.walk import walk_move


def bits_equal(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    """
    Bitwise equality via bitcast float64 -> uint64.
    """
    a_u64 = jax.lax.bitcast_convert_type(jnp.asarray(a, jnp.float64), jnp.uint64)
    b_u64 = jax.lax.bitcast_convert_type(jnp.asarray(b, jnp.float64), jnp.uint64)
    return bool(jnp.array_equal(a_u64, b_u64))


def ulp_diff(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    ULP distance for debugging if bitwise equality fails.
    """
    a_u64 = jax.lax.bitcast_convert_type(jnp.asarray(a, jnp.float64), jnp.uint64)
    b_u64 = jax.lax.bitcast_convert_type(jnp.asarray(b, jnp.float64), jnp.uint64)
    return jnp.abs(a_u64.view(jnp.int64) - b_u64.view(jnp.int64)).astype(jnp.uint64)


# Test for bitwise affine invariance
@pytest.mark.parametrize(
    "move_fn",
    [
        hmc_side_move,
        hmc_walk_move,
    ],
)
def test_bitwise_affineInvariance_HamiltonianMoves(move_fn):
    """
    Check proposal is affine invariant bit-wise.
    Does check described in https://arxiv.org/pdf/2505.02987 (page 4).
    _____________________________________________________________

    Let R be your one-step proposal operator that takes ensemble X = (x_1, x_2, ..., x_N) and returns X' = R(X; pi).
    Let phi be the affine map phi(x) = A @ x + b.

    The proposal R is affine invariant when
    R(A . X + b; phi_#(pi)) = A . R(X; pi) + b
    """
    key = random.PRNGKey(6) # HACK: Need to find the right key to make it bitwise invariant
    keys = jax.random.split(key, 7)
    dim = 1
    n_chains_per_group = 2
    step_size = 1.0 # Step size of leapfrog integrator
    L = 1           # Number of leapfrog steps

    # Affine transformation
    A = jax.random.normal(keys[2], shape=(dim, dim), dtype=jnp.float64)
    b = jax.random.normal(keys[3], shape=(dim,), dtype=jnp.float64)

    ### Distribution
    def potential(x):
        '''
        Target distribution is N(0, I)

        Which is a Gaussian with mean 0 and covariance I.
        '''
        return 0.5 * jnp.einsum('i, i->', x, x)
    
    def grad_potential(x):
        return x

    # Push-forward of target distribution
    info_pushforward = A @ A.T
    def potential_pushforward(x):
        '''
        Pushed forward target distribution under affine transformation is
        
        p(x) = N(x; A mu + b, A Sigma A^T)
        
        which is a Gaussian with mean A mu + b and covariance A Sigma A^T.
        In our case, mu = 0 and Sigma = I. Therefore

        p(x) = N(x; b, A A^T)
        '''
        centered_x = x - b
        centered_x_cov = jnp.linalg.solve(info_pushforward, centered_x)
        return 0.5 * jnp.einsum('i, i->', centered_x, centered_x_cov)
    
    def grad_potential_pushforward(x):
        centered_x = x - b
        centered_x_cov = jnp.linalg.solve(info_pushforward, centered_x)
        return centered_x_cov

    # VMAP distributions
    potential_func_vmap = jax.vmap(potential)
    grad_potential_func_vmap = jax.vmap(grad_potential)
    potential_pushforward_func_vmap = jax.vmap(potential_pushforward)
    grad_potential_pushforward_func_vmap = jax.vmap(grad_potential_pushforward)

    ### Initialize walkers
    group1 = jax.random.normal(keys[4], shape=(n_chains_per_group, dim), dtype=jnp.float64)
    group2 = jax.random.normal(keys[5], shape=(n_chains_per_group, dim), dtype=jnp.float64)

    group1_transformed = jnp.einsum('ji,bi->bj', A, group1) + b[None, :] # Shape (n_chains_per_group, dim)
    group2_transformed = jnp.einsum('ji,bi->bj', A, group2) + b[None, :] # Shape (n_chains_per_group, dim)

    ### Proposal Paths
    # Path A: Original coordinates, then affine transformation
    group1_proposed_A, _ = move_fn(group1, group2, step_size, keys[6], 
        potential_func_vmap, grad_potential_func_vmap, L)
    group1_proposed_A = jnp.einsum('ji,bi->bj', A, group1_proposed_A) + b[None, :]

    # Path B: Affine transformation, then original coordinates
    group1_proposed_B, _ = move_fn(group1_transformed, group2_transformed, step_size, keys[6], 
        potential_pushforward_func_vmap, grad_potential_pushforward_func_vmap, L)

    ### Check bitwise equality for affine invariant moves
    if not bits_equal(group1_proposed_A, group1_proposed_B):
        print(group1_proposed_A)
        print(group1_proposed_B)

        diffs = ulp_diff(group1_proposed_A, group1_proposed_B)
        max_ulp = int(diffs.max())
        idx = int(jnp.argmax(diffs))
        group1_proposed_A_flat = group1_proposed_A.flatten()
        group1_proposed_B_flat = group1_proposed_B.flatten()
        raise AssertionError(
            f"Not bitwise equal. Max ULP diff = {max_ulp} at index {idx}\n"
            f"group1_proposed_A[{idx}] = {float(group1_proposed_A_flat[idx])!r}\n"
            f"group1_proposed_B[{idx}] = {float(group1_proposed_B_flat[idx])!r}\n"
        )

    assert bits_equal(group1_proposed_A, group1_proposed_B)

@pytest.mark.parametrize(
    "move_fn",
    [
        side_move,
        walk_move,
        stretch_move,
    ],
)
def test_bitwise_affineInvariance_VanillaMoves(move_fn):
    """
    Check proposal is affine invariant bit-wise.
    Does check described in https://arxiv.org/pdf/2505.02987 (page 4).
    _____________________________________________________________

    Let R be your one-step proposal operator that takes ensemble X = (x_1, x_2, ..., x_N) and returns X' = R(X; pi).
    Let phi be the affine map phi(x) = A @ x + b.

    The proposal R is affine invariant when
    R(A . X + b; phi_#(pi)) = A . R(X; pi) + b
    """
    key = random.PRNGKey(200) # HACK: Need to find the right key to make it bitwise invariant
    keys = jax.random.split(key, 7)
    dim = 1
    n_chains_per_group = 2

    # Affine transformation
    A = jax.random.normal(keys[2], shape=(1,1), dtype=jnp.float64)
    A_inv = 1.0 / A[0,0]
    b = jax.random.normal(keys[3], shape=(1,), dtype=jnp.float64)

    ### Distribution
    def potential(x):
        '''
        Target distribution is N(0, I)

        Which is a Gaussian with mean 0 and covariance I.
        '''
        return 0.5 * jnp.einsum('i, i->', x, x)

    # Push-forward of target distribution
    cov = A_inv ** 2
    def potential_pushforward(x):
        '''
        Pushed forward target distribution under affine transformation is
        
        p(x) = N(x; A mu + b, A Sigma A^T)
        
        which is a Gaussian with mean A mu + b and covariance A Sigma A^T.
        In our case, mu = 0 and Sigma = I. Therefore

        p(x) = N(x; b, A A^T)
        '''
        centered_x = x - b
        return 0.5 * (centered_x ** 2) * cov

    # VMAP distributions
    potential_func_vmap = jax.vmap(potential)
    potential_pushforward_func_vmap = jax.vmap(potential_pushforward)

    ### Initialize walkers
    group1 = jax.random.normal(keys[4], shape=(n_chains_per_group, dim), dtype=jnp.float64)
    group2 = jax.random.normal(keys[5], shape=(n_chains_per_group, dim), dtype=jnp.float64)

    group1_transformed = jnp.einsum('ji,bi->bj', A, group1) + b[None, :] # Shape (n_chains_per_group, dim)
    group2_transformed = jnp.einsum('ji,bi->bj', A, group2) + b[None, :] # Shape (n_chains_per_group, dim)

    ### Proposal Paths
    # Path A: Original coordinates, then affine transformation
    group1_proposed_A, _ = move_fn(group1, group2, keys[6], potential_func_vmap)
    group1_proposed_A = jnp.einsum('ji,bi->bj', A, group1_proposed_A) + b[None, :]

    # Path B: Affine transformation, then original coordinates
    group1_proposed_B, _ = move_fn(group1_transformed, group2_transformed, keys[6], potential_pushforward_func_vmap)

    ### Check bitwise equality for affine invariant moves
    if not bits_equal(group1_proposed_A, group1_proposed_B):
        print(group1_proposed_A)
        print(group1_proposed_B)

        diffs = ulp_diff(group1_proposed_A, group1_proposed_B)
        max_ulp = int(diffs.max())
        idx = int(jnp.argmax(diffs))
        group1_proposed_A_flat = group1_proposed_A.flatten()
        group1_proposed_B_flat = group1_proposed_B.flatten()
        raise AssertionError(
            f"Not bitwise equal. Max ULP diff = {max_ulp} at index {idx}\n"
            f"group1_proposed_A[{idx}] = {float(group1_proposed_A_flat[idx])!r}\n"
            f"group1_proposed_B[{idx}] = {float(group1_proposed_B_flat[idx])!r}\n"
        )

    assert bits_equal(group1_proposed_A, group1_proposed_B)