Autocorrelation
===============

We include a function to compute the integrated autocorrelation time of a time series. 
This is approximately the number of proposals required to get an independent sample.

This is code modified from `emcee.autocorr <https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py>`_ to handle :code:`JAX`.

.. autofunction:: hemcee.autocorr.integrated_time

.. autofunction:: hemcee.autocorr.function_1d