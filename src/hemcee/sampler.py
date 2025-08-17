"""Minimal emcee-like sampler implemented in JAX."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple


class HamiltonianEnsembleSampler:
    """Simplified affine-invariant sampler.

    This class mimics the API of ``emcee.EnsembleSampler`` but delegates
    the actual work to JAX.  The implementation here is intentionally
    minimal and meant only as boilerplate.
    """

    def __init__(
        self,
        n_walkers_per_group: int,
        ndim: int,
        log_prob_fn: Callable,
        args: Optional[Sequence] = None,
        style: str = "side",
    ) -> None:
        self.nwalkers_per_group = int(n_walkers_per_group)
        self.ndim = int(ndim)
        self.log_prob_fn = log_prob_fn
        self.args = list(args) if args is not None else []
        self.style = style
        self.chain = None
        self.log_prob = None

    def run_mcmc(self, key, initial_state, nsteps: int) -> Tuple["jax.Array", "jax.Array"]:
        """Execute a very small MCMC loop.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key used internally by JAX.
        initial_state : array_like
            Starting position of the walkers.
        nsteps : int
            Number of steps to sample.
        """
        try:
            import jax.numpy as jnp
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency injection
            raise ModuleNotFoundError("jax is required to run the sampler") from exc

        nsteps = int(nsteps)
        initial_state = jnp.asarray(initial_state)
        self.chain = jnp.broadcast_to(initial_state, (nsteps,) + initial_state.shape)
        lp = self.log_prob_fn(initial_state, *self.args)
        self.log_prob = jnp.broadcast_to(lp, (nsteps,))
        return self.chain, self.log_prob

    def sample(self, *args, **kwargs):
        """Alias for :meth:`run_mcmc` to mimic ``emcee``'s API."""
        return self.run_mcmc(*args, **kwargs)
