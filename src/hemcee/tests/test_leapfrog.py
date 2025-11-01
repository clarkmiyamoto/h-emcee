'''
Test leapfrog integrators.

TODO: Add test for affine invariance of the integrator.
'''
import pytest

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hemcee.moves.hamiltonian.hmc_side import leapfrog_side_move
from hemcee.moves.hamiltonian.hmc_walk import leapfrog_walk_move

from hemcee.tests.distribution import make_distributions

seeds = [0, 42] # Test various initlalizations
dim = 5 # Dimension of distribution
distributions = make_distributions(seeds, dim)
n_chains_per_group = dim * 2
step_size = 0.05
L = 30

def is_time_reversal_symmetric(leapfrog, q, p, grad_log_prob, step_size, L, other_q):
    """
    Test time reversal symmetry of the leapfrog integrator.
    That is if you integrate forward, 
    change the momentum, and then integrate backwards, 
    then flip the momentum again--
    you should end up with the same positions and momenta.

    Args:
        leapfrog (Callable): Leapfrog integrator.
        q (jnp.ndarray): Positions.
        p (jnp.ndarray): Momenta.
        grad_log_prob (Callable): Gradient of the log probability.
        step_size (float): Step size.
        L (int): Number of leapfrog steps.
        other_q (jnp.ndarray): Other positions.

    Returns:
        (bool): True if the integrator is time reversal symmetric, False otherwise.
    """
    # Run integrator forwards in time
    qL, pL, pL_projected = leapfrog(q, p, 
        grad_log_prob, step_size, L, other_q)
    
    # Run integrator backwards in time
    qR_in = qL
    pR_in = -1 * pL
    qR, pR, pR_projected = leapfrog(qR_in, pR_in, 
        grad_log_prob, step_size, L, other_q)
    
    qR_out = qR
    pR_out = -1 * pR
    
    # Do you end up where you started?
    return jnp.allclose(q, qR_out) and jnp.allclose(p, pR_out)


# Walk Move
def make_positions_walk(key: jax.random.PRNGKey):
    keys = jax.random.split(key, 3)
    group1 = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    group2 = jax.random.normal(keys[1], shape=(n_chains_per_group, dim))

    centered = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # shape (n_chains_per_group, dim)

    momentum = jax.random.normal(keys[2], shape=(n_chains_per_group, n_chains_per_group))

    return group1, momentum, centered

@pytest.mark.parametrize("log_prob", distributions)
def test_TimeReversalSymmetry_Walk(log_prob):
    '''
    Test time reversal symmetry of the walk leapfrog integrator.

    That is if you integrate forward, 
    change the momentum, and then integrate backwards, 
    then flip the momentum again--
    you should end up with the same positions and momenta.
    '''
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    score_vmap = jax.vmap(jax.grad(log_prob))

    q, p, centered = make_positions_walk(keys[3])

    assert is_time_reversal_symmetric(leapfrog_walk_move, q, p, score_vmap, step_size, L, centered)


# Side Move
def make_positions_side(key: jax.random.PRNGKey):
    keys = jax.random.split(key, 2)
    group1 = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    group2 = jax.random.normal(keys[1], shape=(2, dim))

    difference = (group2[0] - group2[1]) / jnp.sqrt(2*dim)

    momentum = jax.random.normal(keys[2], shape=(n_chains_per_group,))
    return group1, momentum, difference

@pytest.mark.parametrize("log_prob", distributions)
def test_TimeReversalSymmetry_Side(log_prob):
    '''
    Test time reversal symmetry of the side leapfrog integrator.
    '''
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    score_vmap = jax.vmap(jax.grad(log_prob))

    q, p, difference = make_positions_side(keys[3])

    assert is_time_reversal_symmetric(leapfrog_side_move, q, p, score_vmap, step_size, L, difference)
    