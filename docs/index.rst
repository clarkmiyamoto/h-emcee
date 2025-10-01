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

Minimal Example
---------------


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


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/moves



.. _emcee_url: https://github.com/dfm/emcee
.. _ychen25_url: https://arxiv.org/abs/2505.02987

