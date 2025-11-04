import hemcee
from hemcee.backend import Backend
from hemcee.backend import HDFBackend

_all_backends_ = [Backend(), HDFBackend(filename='temp.h5')]

import jax 
import jax.numpy as jnp

import pytest



total_chains = 4
dim = 1
num_samples = 10
warmup = 35
thin_by = 2

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)

initial_state = jax.random.normal(keys[0], shape=(total_chains, dim))

def log_prob(x):
    return -0.5 * jnp.einsum('i,i ->', x, x)

@pytest.mark.parametrize('backend', _all_backends_)
def test_iterations(backend: Backend):

    sampler = hemcee.HamiltonianEnsembleSampler(total_chains, dim, log_prob, backend=backend)
    samples = sampler.run_mcmc(keys[0], initial_state, num_samples=num_samples, warmup=warmup, thin_by=thin_by)

    assert samples.shape == (num_samples, total_chains, dim), 'Outputted shape is wrong'

    assert sampler.backend.iteration == (warmup + num_samples * thin_by), 'Iteration count is wrong'


