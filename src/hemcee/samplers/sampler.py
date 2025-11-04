"""
High-level sampler implementations for hemcee.
"""

from abc import ABC
from typing import Callable
import jax
import jax.numpy as jnp
import hemcee
from hemcee.backend.backend import Backend

class BaseSampler(ABC):
    """Base class for MCMC samplers with common functionality.

    Attributes:
        total_chains (int): Total number of ensemble chains.
        dim (int): Dimensionality of the target distribution.
        log_prob (Callable): Vectorized log-probability function.
        grad_log_prob (Callable): Vectorized gradient of the log probability.
        move (Callable): Proposal function updating each ensemble group.
        backend (Backend): Backend for storing chain data.
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        move: Callable,
        backend: Backend = None,
    ) -> None:
        """Initialize the base sampler configuration.

        Args:
            total_chains (int): Total number of chains in the ensemble.
            dim (int): Dimensionality of the target distribution.
            log_prob (Callable): Callable returning the log density of the
                target distribution (doesn't need to be normalized).
            move (Callable): Proposal function used for ensemble updates.
            backend: Backend for storing chain data.
        """
        if total_chains < 4:
            raise ValueError("`total_chains` must be at least 4 to form meaningful ensemble groups")
        
        self.total_chains = int(total_chains)
        self.dim = int(dim)
        self.move = move
        self.backend = Backend() if backend is None else backend
        self.backend.reset(total_chains, dim)

        # JIT and Vectorize log prob.
        self.log_prob = jax.jit(jax.vmap(log_prob))

        # Validate gradient output as well
        self.grad_log_prob = jax.jit(jax.vmap(jax.grad(log_prob)))
        
        # Initialize diagnostics placeholders
        self.diagnostics_warmup = None
        self.diagnostics_main = None

    def _validate_mcmc_inputs(self, 
                              num_samples: int, 
                              warmup: int, 
                              thin_by: int,
                              initial_state: jnp.ndarray) -> None:
        """Validate common MCMC input parameters.

        Args:
            warmup (int): Number of warmup iterations.
            num_samples (int): Number of post-warmup samples to retain.
            thin_by (int): Keep every `thin_by` sample.
        """
        if thin_by < 1:
            raise ValueError("`thin_by` must be 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")

        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")
        
        if initial_state.shape != (self.total_chains, self.dim):
            raise ValueError('`inital_state` needs to have shape ')

    def get_chain(self, 
                  discard: int = 0, 
                  thin: int = 1, 
                  flat: bool = False) -> jnp.ndarray:
        return self.backend.get_chain(discard=discard, thin=thin, flat=flat)
    
    def get_logprob(self,
                    discard: int = 0, 
                    thin: int = 1, 
                    flat: bool = False) -> jnp.ndarray:
        return self.backend.get_log_prob(discard=discard, thin=thin, flat=flat)
    
    def get_acceptance_prob(self):
        return self.backend.accepted / self.backend.iteration
    
    def get_autocorr(self, discard, thin):
        x = self.get_chain(discard=discard, thin=thin, flat=False)
        return hemcee.autocorr.integrated_time(x)