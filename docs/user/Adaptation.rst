Adaptive Hyper-parameters
=========================

We include affine invariant schemes for tuning hyper-parameters 
of HMC samplers (e.g. step size, number of leapfrog steps). 
Adaptation schemes are children of the Adapter class

.. autoclass:: hemcee.adaptation.base.Adapter

Global Adaptation
-----------------
These schemes set hyperparameters globally, that is they don't change after warmup.

Dual Averaging
~~~~~~~~~~~~~~
Adjust the step size to a get a target acceptance rate. 
See https://arxiv.org/abs/1111.4246.

.. autoclass:: hemcee.adaptation.dual_averaging.DAParameters

.. autoclass:: hemcee.adaptation.dual_averaging.DualAveragingAdapter

ChEES
~~~~~
Adjusts integration length by looking at the Change in the Estimator of 
the Expected Square of the parameter. It sets the length statically for the
entire MCMC run. See https://proceedings.mlr.press/v130/hoffman21a.html.

.. math::
   :label: eq:chees_def

   \mathrm{ChEES}
   \;=\; \frac{1}{4}\,
   \mathbb{E}\!\left[\left(\|\theta' - \mathbb{E}[\theta]\|^2
   - \|\theta - \mathbb{E}[\theta]\|^2 \right)^2\right].

where :math:`\theta'` is the post-leapfrog state (i.e.
:math:`\theta',r'=\mathrm{leapfrog}_{\varepsilon,L}(\theta,r)`).

.. autoclass:: hemcee.adaptation.chees.ChEESParameters

.. autoclass:: hemcee.adaptation.chees.ChEESAdapter

Local Adaptation
----------------
These schemes set hyperparameters locally, that is they parameters per proposal step.

NUTS
~~~~
Adjust step size via the No-U-Turn condition. It sets the length locally 
(meaning per MCMC proposal). See https://arxiv.org/abs/1111.4246.
We also have an modification to the No-U-Turn condition which makes the algorithm affine invariant.





