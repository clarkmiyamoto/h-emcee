import jax
import jax.numpy as jnp
from jax.scipy.special import erf

import hemcee

def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

def _test_normal(
    proposal,
    ndim=1,
    nwalkers=32,
    nsteps=2000,
    seed=1234,
):
    key = jax.random.PRNGKey(seed)

    init = jax.random.normal(keys[0], (nwalkers, ndim))

    sampler = hemcee.HamiltonianEnsembleSampler(
        total_chains=nwalkers,
        dim=ndim,
        log_prob=lambda x: -0.5 * jnp.sum(x**2),
        move=proposal,
    )
    samples, diagnostics = sampler.run_mcmc(key, init, nsteps)
    



