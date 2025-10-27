import pytest
import jax
import jax.numpy as jnp

import hemcee
from hemcee.adaptation.dual_averaging import DAParameters, DualAveragingAdapter
from hemcee.adaptation.chees import ChEESParameters, ChEESAdapter
from hemcee.adaptation.adapter import Adapter, NoOpAdapter, CompositeAdapter


seed = 0
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, num=2)

n_walkers = 4
n_dim = 1

def log_prob(x):
    return -0.5 * jnp.sum(x**2)

inital_state = jax.random.normal(keys[0], shape=(n_walkers, n_dim))

def test_default_adaptation_both_enabled():
    """
    Test default behavior of sampler, ensure dual averaging & ChEES is auto-enabled.
    """
    sampler_Default = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    samples_fromDefault = sampler_Default.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5)

    sampler_BothEnabled = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    samples_fromBothEnabled = sampler_BothEnabled.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5, 
                                                           adapt_step_size=True , adapt_length=True)

    assert isinstance(sampler_Default.adapter, CompositeAdapter), "Default adapter should be CompositeAdapter"
    assert isinstance(sampler_BothEnabled.adapter, CompositeAdapter), "Both enabled adapter should be CompositeAdapter"
    assert jnp.allclose(samples_fromDefault, samples_fromBothEnabled), "Outputs were not the same"

def test_stepsize_adaptation_disabled():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5, adapt_step_size=False , adapt_length=True)

    assert isinstance(sampler.adapter, ChEESAdapter), 'Adapter is not `ChEESAdapter`'

def test_chees_adaptation_disable():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(keys[1], inital_state,  num_samples=1, warmup=5, adapt_step_size=True , adapt_length=False, )

    assert isinstance(sampler.adapter, DualAveragingAdapter), 'Adapter is not `DualAveragingAdapter`'

def test_no_adaptation():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5, adapt_step_size=False, adapt_length=False)

    assert isinstance(sampler.adapter, NoOpAdapter), 'Adapter is not `NoOpAdapter`'




