'''
Test leapfrog integrators.
'''
import pytest

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hemcee.moves.hamiltonian.hmc_side import leapfrog_side_move as hmc_side_leapfrog
from hemcee.moves.hamiltonian.hmc_walk import leapfrog_walk_move as hmc_walk_leapfrog


def grad_potential(x):
    '''
    Gradient of unit gaussian.
    '''
    return x


def test_TimeReversalSymmetry_Walk():
    '''
    Test time reversal symmetry of the walk leapfrog integrator.
    '''
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 3)
    dim = 1
    n_chains_per_group = dim * 2
    step_size = 0.1
    L = 2

    grad_potential_func_vmap = jax.vmap(grad_potential)

    q = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    other_q = jax.random.normal(keys[1], shape=(n_chains_per_group, dim))
    p = jax.random.normal(keys[2], shape=(n_chains_per_group, n_chains_per_group))

    # Run integrator forwards in time
    qL, pL = hmc_walk_leapfrog(q, p, 
        grad_potential_func_vmap, step_size, L, other_q)
    
    # Run integrator backwards in time
    qR_in = qL
    pR_in = -1 * pL
    qR, pR = hmc_walk_leapfrog(qR_in, pR_in, 
        grad_potential_func_vmap, step_size, L, other_q)
    
    qR_out = qR
    pR_out = -1 * pR
    
    # Do you end up where you started?
    print(q - qR_out)
    print(p - pR_out)
    assert jnp.array_equal(q, qR_out)
    assert jnp.array_equal(p, pR_out)

def test_TimeReversalSymmetry_Side():
    '''
    Test time reversal symmetry of the side leapfrog integrator.
    '''
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 3)
    dim = 1
    n_chains_per_group = dim * 2
    step_size = 0.1
    L = 2

    grad_potential_func_vmap = jax.vmap(grad_potential)

    q = jax.random.normal(keys[0], shape=(n_chains_per_group, dim))
    other_q = jax.random.normal(keys[1], shape=(n_chains_per_group, dim))
    p = jax.random.normal(keys[2], shape=(n_chains_per_group,))

    # Run integrator forwards in time
    qL, pL = hmc_side_leapfrog(q, p, 
        grad_potential_func_vmap, step_size, L, other_q)
    
    # Run integrator backwards in time
    qR_in = qL
    pR_in = -1 * pL
    qR, pR = hmc_side_leapfrog(qR_in, pR_in, 
        grad_potential_func_vmap, step_size, L, other_q)
    
    qR_out = qR
    pR_out = -1 * pR
    
    # Do you end up where you started?
    print(q - qR_out)
    print(p - pR_out)
    assert jnp.array_equal(q, qR_out)
    assert jnp.array_equal(p, pR_out)
    