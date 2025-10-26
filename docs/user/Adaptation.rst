Adaptive Hyper-parameters
=========================

We include affine invariant schemes for tuning hyper-parameters 
of HMC samplers (e.g. step size, number of leapfrog steps). 

Step Size Tuning
----------------

Dual Averaging
~~~~~~~~~~~~~~
Adjust the step size to a get a target acceptance rate. 
See https://arxiv.org/abs/1111.4246.


.. autoclass:: hemcee.adaptation.dual_averaging.DAState

.. autoclass:: hemcee.adaptation.dual_averaging.DAParameters

.. autoclass:: hemcee.adaptation.dual_averaging.da_cond_update
   :members:


Integration Length Tuning
-------------------------

NUTS
~~~~
Adjust step size via the No-U-Turn condition. It sets the length locally 
(meaning per MCMC proposal). See https://arxiv.org/abs/1111.4246.

ChEES
~~~~~
Adjusts integration length by looking at the Change in the Estimator of 
the Expected Square of the parameter. It sets the length statically for the
entire MCMC run. See https://proceedings.mlr.press/v130/hoffman21a.html.


