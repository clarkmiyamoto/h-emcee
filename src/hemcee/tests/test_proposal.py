'''
TODO: Add plots to convince others that you actually can sample the distribution.
TODO: Make all tests independnet of seed.
'''
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from scipy.stats import kstest

import numpy as np
import pytest

import hemcee
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.vanilla.side import side_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.moves.vanilla.stretch import stretch_move

hamiltonian_moves = [hmc_walk_move, hmc_side_move]
vanilla_moves = [walk_move, side_move, stretch_move]


@pytest.mark.parametrize("proposal", [
    hmc_walk_move, 
    hmc_side_move, 
    walk_move, 
    side_move, 
    stretch_move
])
def test_normal(
    proposal,
    ndim=1,
    nwalkers=32,
    nsteps=2000,
    seed=1234,
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)

    init = jax.random.normal(keys[0], (nwalkers, ndim))

    if proposal in hamiltonian_moves:
        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains=nwalkers,
            dim=ndim,
            log_prob=lambda x: - 0.5 * jnp.sum(x**2),
            move=proposal,
        )
    elif proposal in vanilla_moves:
        sampler = hemcee.EnsembleSampler(
            total_chains=nwalkers,
            dim=ndim,
            log_prob=lambda x: - 0.5 * jnp.sum(x**2),
            move=proposal,
        )
    else:
        raise ValueError(f"Invalid proposal: {proposal}")

    samples, diagnostics = sampler.run_mcmc(keys[1], init, nsteps)
    samples = samples.reshape(-1, ndim)

    # samples = np.array(samples).reshape(-1, ndim)

    mu, sig = np.mean(samples, axis=0), np.std(samples, axis=0)

    assert np.all(np.abs(mu) < 0.08), "Incorrect mean"
    assert np.all(np.abs(sig - 1) < 0.05), "Incorrect standard deviation"

    ks_stat, _ = kstest(samples.reshape(-1), 'norm')
    
    assert ks_stat < 0.05, "The K-S test failed"

    



