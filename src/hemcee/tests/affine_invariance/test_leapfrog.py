import pytest
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hemcee.moves.hamiltonian.hmc_walk import leapfrog_walk_move
from hemcee.moves.hamiltonian.hmc_side import leapfrog_side_move

from hemcee.tests.affine_invariance.tools import make_distributions_and_affine_transformation

dim = 5
n_chains_per_group = dim * 4

def make_positions_walk(key: jax.random.PRNGKey):
    keys = jax.random.split(key, 3)
    group1 = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    group2 = jax.random.normal(keys[1], shape=(n_chains_per_group, dim))

    centered = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # shape (n_chains_per_group, dim)

    momentum = jax.random.normal(keys[2], shape=(n_chains_per_group, n_chains_per_group))

    return group1, momentum, centered

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_AffineInvariance_Walk(seed: int):
    '''
    Test affine invariance of the walk leapfrog integrator.
    '''
    # Integration parametres
    step_size = 0.1
    L = 10

    # Make distributions and affine transformation
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    log_prob, log_prob_pushforward, A, b = make_distributions_and_affine_transformation(keys[2], dim)

    score_vmap = jax.vmap(jax.grad(log_prob))
    score_vmap_pushforward = jax.vmap(jax.grad(log_prob_pushforward))

    # Make samples
    q, p, centered = make_positions_walk(keys[3])

    q_transformed = jnp.einsum('ij,bj->bi', A, q) + b[None, :]
    centered_transformed = centered @ A.T # # Shape (n_chains_per_group, dim)

    # Run leapfrog integrator
    q_regular, p_regular, p_projected = leapfrog_walk_move(q, p, score_vmap, step_size, L, centered)
    q_regular = jnp.einsum('ij,bj->bi', A, q_regular) + b[None, :]
    q_transformed, p_transformed, p_projected_transformed = leapfrog_walk_move(q_transformed, p, score_vmap_pushforward, step_size, L, centered_transformed)

    # Check equality for affine invariant moves
    assert jnp.allclose(q_transformed, q_regular)
    assert jnp.allclose(p_transformed, p_regular)

def make_positions_side(key: jax.random.PRNGKey):
    keys = jax.random.split(key, 3)
    group1 = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    group2 = jax.random.normal(keys[1], shape=(2, dim))

    centered = (group2[0] - group2[1]) / jnp.sqrt(2*dim) # Shape (dim,)

    momentum = jax.random.normal(keys[2], shape=(1,))

    return group1, momentum, centered

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_AffineInvariance_Side(seed: int):
    '''
    Test affine invariance of the side leapfrog integrator.
    '''
    # Integration parametres
    step_size = 0.1
    L = 10

    # Make distributions and affine transformation
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    log_prob, log_prob_pushforward, A, b = make_distributions_and_affine_transformation(keys[2], dim)

    score_vmap = jax.vmap(jax.grad(log_prob))
    score_vmap_pushforward = jax.vmap(jax.grad(log_prob_pushforward))

    # Make samples
    q, p, centered = make_positions_side(keys[3])

    q_transformed = jnp.einsum('ij,bj->bi', A, q) + b[None, :]
    centered_transformed = centered @ A.T  # Shape (dim,)

    # Run leapfrog integrator
    q_regular, p_regular, p_projected = leapfrog_side_move(q, p, score_vmap, step_size, L, centered)
    q_regular = jnp.einsum('ij,bj->bi', A, q_regular) + b[None, :]
    q_transformed, p_transformed, p_projected_transformed = leapfrog_side_move(q_transformed, p, score_vmap_pushforward, step_size, L, centered_transformed)

    # Check equality for affine invariant moves
    assert jnp.allclose(q_transformed, q_regular)
    assert jnp.allclose(p_transformed, p_regular)
