.. hemcee documentation master file, created by
   sphinx-quickstart on Thu Sep 25 00:53:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`h-emcee` Documentation
=======================

:code:`h-emcee` is an implementation of ensemble Hamiltonian Monte Carlo (HMC) samplers with affine invariance,
based on `Y. Chen (2025) <ychen25_url>`_ and implemented in :code:`JAX`.
We also include affine-invariant schemes for tuning hyperparameters of HMC samplers (e.g., step size, number of leapfrog steps).

The philosophy and syntax of this package are meant as a minimal replacement for `emcee <emcee_url>`_.
This is a pure-Python implementation designed for easy statistical inference (no graphical models needed).

Basic Example
---------------
All you need is access to the unnormalized log probability!

.. code-block:: python
   
   import jax
   import jax.numpy as jnp
   import hemcee

   def log_prob(x):
      return -0.5 * jnp.sum(x ** 2)
   
   key = jax.random.PRNGKey(0)
   keys = jax.random.split(key, 2)

   num_walkers, dim = 100, 5
   inital_states = jax.random.normal(keys[0], shape=(num_walkers, dim))

   sampler = hemcee.HamiltonianEnsembleSampler(num_walkers, dim, log_prob)
   sampler.run_mcmc(keys[1], inital_states, 10000)

For a more through example, see :doc:`tutorials/quickstart`.

Navigating the documentation
----------------------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user/Installation
   user/HamiltonianEnsembleSampler
   user/EnsembleSampler
   user/Adaptation
   user/Autocorrelation
   user/FAQ


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/moves
   tutorials/adaptation



.. _emcee_url: https://github.com/dfm/emcee
.. _ychen25_url: https://arxiv.org/abs/2505.02987

