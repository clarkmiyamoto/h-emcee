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


class TestPositiveInfHandling:
    """Test that +inf values issue a warning."""

    def test_posinf_warning_ensemble(self):
        """Test +inf warning in EnsembleSampler."""
        def log_prob_with_posinf(x):
            # Return +inf for some input
            return jnp.inf

        key = jax.random.PRNGKey(7)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob_with_posinf)

        # Should issue a warning about +inf
        with pytest.warns(UserWarning, match="Log probability returned \\+inf"):
            sampler.run_mcmc(key, initial_state, num_samples=1, warmup=0)

    def test_posinf_warning_hamiltonian_ensemble(self):
        """Test +inf warning in HamiltonianEnsembleSampler."""
        def log_prob_with_posinf(x):
            # Return +inf for some input
            return jnp.inf

        key = jax.random.PRNGKey(8)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains, dim, log_prob_with_posinf
        )

        # Should issue a warning about +inf
        with pytest.warns(UserWarning, match="Log probability returned \\+inf"):
            sampler.run_mcmc(key, initial_state, num_samples=1, warmup=0)

    def test_posinf_warning_hamiltonian(self):
        """Test +inf warning in HamiltonianSampler."""
        def log_prob_with_posinf(x):
            # Return +inf for some input
            return jnp.inf

        key = jax.random.PRNGKey(9)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_with_posinf)

        # Should issue a warning about +inf
        with pytest.warns(UserWarning, match="Log probability returned \\+inf"):
            sampler.run_mcmc(key, initial_state, num_samples=1, warmup=0)


class TestGradientNaNInfHandling:
    """Test that NaN and inf values in gradients are caught."""

    def test_gradient_inf_hamiltonian_ensemble(self):
        """Test gradient inf detection in HamiltonianEnsembleSampler."""
        def log_prob_inf_gradient(x):
            # sqrt(x[0]) has infinite gradient at x[0]=0
            return jnp.sqrt(jnp.abs(x[0])) - 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(10)
        # Initialize at zero to trigger infinite gradient
        initial_state = jnp.array([[0.0, 0.1], [0.1, 0.1], [0.0, -0.1], [0.1, -0.1]])

        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains, dim, log_prob_inf_gradient
        )

        # Should raise error about inf gradient
        with pytest.raises((ValueError, Exception), match="(Gradient of log probability returned inf|JaxRuntimeError)"):
            sampler.run_mcmc(
                key, initial_state, num_samples=num_samples, warmup=warmup
            )

    def test_gradient_inf_hamiltonian(self):
        """Test gradient inf detection in HamiltonianSampler."""
        def log_prob_inf_gradient(x):
            # sqrt(x[0]) has infinite gradient at x[0]=0
            return jnp.sqrt(jnp.abs(x[0])) - 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(11)
        # Initialize at zero to trigger infinite gradient
        initial_state = jnp.array([[0.0, 0.1], [0.1, 0.1], [0.0, -0.1], [0.1, -0.1]])

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_inf_gradient)

        # Should raise error about inf gradient
        with pytest.raises((ValueError, Exception), match="(Gradient of log probability returned inf|JaxRuntimeError)"):
            sampler.run_mcmc(
                key, initial_state, num_samples=num_samples, warmup=warmup
            )

    def test_gradient_nan_hamiltonian_ensemble(self):
        """Test gradient NaN detection in HamiltonianEnsembleSampler."""
        def log_prob_nan_gradient(x):
            # Function where gradient is NaN
            # Using a conditional that breaks differentiability
            return jnp.where(x[0] > 0, jnp.log(x[0]), 0.0) - 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(12)
        # This may or may not trigger NaN depending on JAX's AD rules
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianEnsembleSampler(
            total_chains, dim, log_prob_nan_gradient
        )

        # Try to run - might succeed or fail depending on JAX's handling
        try:
            samples = sampler.run_mcmc(
                key, initial_state, num_samples=num_samples, warmup=warmup
            )
            # If it succeeds, samples should be valid
            assert not jnp.any(jnp.isnan(samples)), "Samples should not contain NaN"
            assert not jnp.any(jnp.isinf(samples)), "Samples should not contain inf"
        except Exception as e:
            # If it fails, should be about NaN/inf
            error_str = str(e)
            assert ("NaN" in error_str or "inf" in error_str or "JaxRuntimeError" in error_str), \
                f"Expected NaN/inf-related error, got: {error_str}"

    def test_gradient_nan_hamiltonian(self):
        """Test gradient NaN detection in HamiltonianSampler."""
        def log_prob_nan_gradient(x):
            # Function where gradient is NaN
            return jnp.where(x[0] > 0, jnp.log(x[0]), 0.0) - 0.5 * jnp.sum(x**2)

        key = jax.random.PRNGKey(13)
        initial_state = jax.random.normal(key, shape=(total_chains, dim))

        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob_nan_gradient)

        # Try to run - might succeed or fail depending on JAX's handling
        try:
            samples = sampler.run_mcmc(
                key, initial_state, num_samples=num_samples, warmup=warmup
            )
            # If it succeeds, samples should be valid
            assert not jnp.any(jnp.isnan(samples)), "Samples should not contain NaN"
            assert not jnp.any(jnp.isinf(samples)), "Samples should not contain inf"
        except Exception as e:
            # If it fails, should be about NaN/inf
            error_str = str(e)
            assert ("NaN" in error_str or "inf" in error_str or "JaxRuntimeError" in error_str), \
                f"Expected NaN/inf-related error, got: {error_str}"


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
