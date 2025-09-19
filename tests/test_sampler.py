import jax
import jax.numpy as jnp
import pytest

from hemcee.sampler import HamiltonianEnsembleSampler


def gaussian_log_prob(x):
    return -0.5 * jnp.dot(x, x)


def test_sampler_requires_minimum_chains():
    with pytest.raises(ValueError):
        HamiltonianEnsembleSampler(total_chains=2, dim=2, log_prob=gaussian_log_prob)


def test_hamiltonian_sampler_shapes_and_diagnostics():
    total_chains = 4
    dim = 2
    sampler = HamiltonianEnsembleSampler(
        total_chains=total_chains,
        dim=dim,
        log_prob=gaussian_log_prob,
        step_size=0.1,
        L=3,
    )

    key = jax.random.PRNGKey(0)
    initial_state = jnp.zeros((total_chains, dim))

    num_samples = 5
    warmup = 3
    thin_by = 2

    samples, diagnostics = sampler.run_mcmc(
        key,
        initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,
        adapt_step_size=True,
        adapt_integration=False,
        show_progress=False,
    )

    assert samples.shape == (num_samples, total_chains, dim)
    assert 'acceptance_rate' in diagnostics
    assert 'step_size' in diagnostics
    assert 'dual_averaging_state' in diagnostics

    acceptance_rate = diagnostics['acceptance_rate']
    assert 0.0 <= float(acceptance_rate) <= 1.0

    step_size = diagnostics['step_size']
    assert float(step_size) > 0.0

    da_state = diagnostics['dual_averaging_state']
    assert float(da_state.step_size) > 0.0
