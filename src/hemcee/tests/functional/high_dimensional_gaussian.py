import hemcee
from hemcee.tests.distribution import make_gaussian, _make_covariance_skewed

import jax
import jax.numpy as jnp
import numpy as np

import corner

# RNG Settings
seed = 0

# Distribution Settings
dim = 100
condition_number = 1000

# Sampler Settings
total_chains = dim * 2
step_size: float = 0.1 # Step size of leapfrog integrator
L: int = 10  # Number of leapfrog steps


if __name__ == "__main__":
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    true_mean = jax.random.normal(keys[1], shape=(dim,))
    true_precision = _make_covariance_skewed(keys[0], dim, condition_number)
    
    log_prob = make_gaussian(true_precision, true_mean)

    sampler = hemcee.sampler.HamiltonianSampler(
        total_chains=total_chains,
        dim=dim,
        log_prob=log_prob,
        step_size=step_size,
        L=L,
    )

    initial_state = jax.random.normal(keys[1], shape=(total_chains, dim))
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=10000,
        warmup=0,
        adapt_step_size=True,
        thin_by=10,)
    

    print('Diagonstics:')
    acceptance_rate_warmup = sampler.diagnostics_warmup['acceptance_rate']
    acceptance_rate_main = sampler.diagnostics_main['acceptance_rate']

    print(f'  Average warmup acceptance rate: {jnp.mean(acceptance_rate_warmup):.3f}')
    print(f'  Average main acceptance rate: {jnp.mean(acceptance_rate_main):.3f}')

    print('Autocorrelation:')
    tau = hemcee.autocorr.integrated_time(samples)
    print(f'  Integrated autocorrelation: {tau:.3f}')
    
    print('Truth vs Empirical:')
    empirical_mean = jnp.mean(samples, axis=(0,1))
    empirical_cov = jnp.cov(samples.reshape(-1, dim).T)
    
    mean_residual = jnp.linalg.norm(true_mean - empirical_mean)
    cov_residual = jnp.linalg.norm(jnp.linalg.inv(true_precision) - empirical_cov)

    print(f"  Mean residual: {mean_residual}")
    print(f"  Covariance residual: {cov_residual}")
    
    
    
    