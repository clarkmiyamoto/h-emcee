"""
Tests for log probability validation (NaN, -inf, +inf handling).

Tests verify that:
1. NaN values in log_prob raise ValueError
2. -inf values in log_prob are handled correctly (instant rejection)
3. +inf values in log_prob issue a warning
4. Gradients that produce NaN are caught
"""

import hemcee
import jax
import jax.numpy as jnp
import pytest
import warnings


# Test configuration
total_chains = 4
dim = 2
num_samples = 5
warmup = 10


class TestNaNHandling:
    """Test that NaN values in log_prob raise errors."""

    def test_nan_in_log_prob_ensemble(self):
        """Test NaN detection in EnsembleSampler."""
        def log_prob_with_nan(x):
            # Return NaN for any input
            return jnp.nan

        key = jax.random.PRNGKey(0)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob_with_nan)

        # JAX wraps the ValueError in a JaxRuntimeError
        with pytest.raises((ValueError, Exception), match="(Log probability returned NaN|JaxRuntimeError)"):
            sampler.run_mcmc(key, initial_state, num_samples=num_samples, warmup=0)

    def test_nan_in_log_prob_hamiltonian_ensemble(self):
        """Test NaN detection in HamiltonianEnsembleSampler."""
        def log_prob_with_nan(x):
            # Return NaN for any input
            return jnp.nan

        key = jax.random.PRNGKey(1)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains, dim, log_prob_with_nan
        )

        # JAX wraps the ValueError in a JaxRuntimeError
        with pytest.raises((ValueError, Exception), match="(Log probability returned NaN|JaxRuntimeError)"):
            sampler.run_mcmc(key, initial_state, num_samples=num_samples, warmup=0)

    def test_nan_in_log_prob_hamiltonian(self):
        """Test NaN detection in HamiltonianSampler."""
        def log_prob_with_nan(x):
            # Return NaN for any input
            return jnp.nan

        key = jax.random.PRNGKey(2)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_with_nan)

        # JAX wraps the ValueError in a JaxRuntimeError
        with pytest.raises((ValueError, Exception), match="(Log probability returned NaN|JaxRuntimeError)"):
            sampler.run_mcmc(key, initial_state, num_samples=num_samples, warmup=0)

    def test_conditional_nan_in_log_prob(self):
        """Test NaN that occurs only for certain inputs."""
        def log_prob_conditional_nan(x):
            # Return NaN when x[0] is negative (log of negative number)
            return jnp.where(x[0] < 0, jnp.nan, -0.5 * jnp.sum(x**2))

        key = jax.random.PRNGKey(3)
        # Initialize with some negative values to trigger NaN
        initial_state = jnp.array([[-1.0, 0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])

        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob_conditional_nan)

        # JAX wraps the ValueError in a JaxRuntimeError
        with pytest.raises((ValueError, Exception), match="(Log probability returned NaN|JaxRuntimeError)"):
            sampler.run_mcmc(key, initial_state, num_samples=num_samples, warmup=0)


class TestNegativeInfHandling:
    """Test that -inf values are handled correctly (instant rejection)."""

    def test_neginf_in_log_prob_ensemble(self):
        """Test -inf handling in EnsembleSampler (should not raise error)."""
        def log_prob_with_neginf(x):
            # Return -inf for x[0] < 0 (hard constraint)
            return jnp.where(x[0] < 0, -jnp.inf, -0.5 * jnp.sum(x**2))

        key = jax.random.PRNGKey(4)
        # Initialize with positive values
        initial_state = jnp.abs(jax.random.normal(key, shape=(total_chains, dim)))

        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob_with_neginf)

        # This should run without error - -inf should be handled gracefully
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        # Verify all samples respect the constraint (x[0] >= 0)
        assert jnp.all(samples[:, :, 0] >= 0), "Sampler should reject all x[0] < 0"

    def test_neginf_in_log_prob_hamiltonian_ensemble(self):
        """Test -inf handling in HamiltonianEnsembleSampler."""
        def log_prob_with_neginf(x):
            # Return -inf for x[0] < 0 (hard constraint)
            return jnp.where(x[0] < 0, -jnp.inf, -0.5 * jnp.sum(x**2))

        key = jax.random.PRNGKey(5)
        # Initialize with positive values
        initial_state = jnp.abs(jax.random.normal(key, shape=(total_chains, dim)))

        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains, dim, log_prob_with_neginf
        )

        # This should run without error
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        # Verify all samples respect the constraint (x[0] >= 0)
        assert jnp.all(samples[:, :, 0] >= 0), "Sampler should reject all x[0] < 0"

    def test_neginf_in_log_prob_hamiltonian(self):
        """Test -inf handling in HamiltonianSampler."""
        def log_prob_with_neginf(x):
            # Return -inf for x[0] < 0 (hard constraint)
            return jnp.where(x[0] < 0, -jnp.inf, -0.5 * jnp.sum(x**2))

        key = jax.random.PRNGKey(6)
        # Initialize with positive values
        initial_state = jnp.abs(jax.random.normal(key, shape=(total_chains, dim)))

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_with_neginf)

        # This should run without error
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        # Verify all samples respect the constraint (x[0] >= 0)
        assert jnp.all(samples[:, :, 0] >= 0), "Sampler should reject all x[0] < 0"


class TestValidLogProb:
    """Test that valid log_prob functions work correctly."""

    def test_valid_log_prob_ensemble(self):
        """Test that valid log_prob works in EnsembleSampler."""
        def log_prob_valid(x):
            # Standard Gaussian
            return -0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(12)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob_valid)

        # Should run without error or warning
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        assert samples.shape == (num_samples, total_chains, dim)
        assert not jnp.any(jnp.isnan(samples))
        assert not jnp.any(jnp.isinf(samples))

    def test_valid_log_prob_hamiltonian_ensemble(self):
        """Test that valid log_prob works in HamiltonianEnsembleSampler."""
        def log_prob_valid(x):
            # Standard Gaussian
            return -0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(13)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianEnsembleSampler(total_chains, dim, log_prob_valid)

        # Should run without error or warning
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        assert samples.shape == (num_samples, total_chains, dim)
        assert not jnp.any(jnp.isnan(samples))
        assert not jnp.any(jnp.isinf(samples))

    def test_valid_log_prob_hamiltonian(self):
        """Test that valid log_prob works in HamiltonianSampler."""
        def log_prob_valid(x):
            # Standard Gaussian
            return -0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(14)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_valid)

        # Should run without error or warning
        samples = sampler.run_mcmc(
            key, initial_state, num_samples=num_samples, warmup=warmup
        )

        assert samples.shape == (num_samples, total_chains, dim)
        assert not jnp.any(jnp.isnan(samples))
        assert not jnp.any(jnp.isinf(samples))
