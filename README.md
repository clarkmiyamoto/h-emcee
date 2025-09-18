# h-emcee
Implementation of affine invariant Hamiltonian samplers [Y. Chen (2025)](https://arxiv.org/abs/2505.02987) in JAX + affine invariant schemes for tuning hyper-parameters for Hamiltonian samplers. The syntax of this package is meant as a minimal replacement for [emcee](https://github.com/dfm/emcee).

# Installation
```
pip install h-emcee
```

# Simple Usage
```
import hemcee
import jax
import jax.numpy as jnp

def log_prob(x, ivar):
    covariance = ivar[0]
    return -0.5 * jnp.einsum('i,ij,j->', centered, covariance, centered)

key = jax.random.PRNGKey(0)
ndim, n_walkers_per_group = 5, 100
num_samples = 1e5
ivar = [jnp.eye(dim)]
inital = np.random.randn(nwalkers, ndim)

sampler = hemcee.HamiltonianEnsembleSampler(n_walkers_per_group, ndim, log_prob, args=[ivar], style='side')
samples, diagnostics = sampler.run_mcmc(key, inital, num_samples)
```

# Todo
[ ] Implement affine invariant ChEES algorithm. This is better than NUTs (interms of control flow).