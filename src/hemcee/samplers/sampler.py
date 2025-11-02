"""
High-level sampler implementations for hemcee.
"""

from abc import ABC
from typing import Callable
import jax
import jax.numpy as jnp
import warnings
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
        validated_log_prob = _validate_log_prob_output(log_prob) # Raises error if outputting NaNs
        self.log_prob = jax.jit(jax.vmap(validated_log_prob))

        # Validate gradient output as well
        validated_grad_log_prob = _validate_grad_log_prob_output(log_prob)
        self.grad_log_prob = jax.jit(jax.vmap(validated_grad_log_prob))
        
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
    
def _validate_log_prob_output(log_prob_fn: Callable) -> Callable:
    """Wrap log_prob function with NaN/inf validation.

    This wrapper checks the output of the log probability function for:
    - NaN values: Raises a ValueError as these indicate bugs in the log_prob function
    - +inf values: Issues a warning as these are suspicious (infinite probability)
    - -inf values: Allowed (represents zero probability, will be rejected naturally)

    Args:
        log_prob_fn (Callable): The log probability function to validate.

    Returns:
        Callable: Validated log probability function.

    Raises:
        ValueError: If the log probability returns NaN.
    """
    def validated_fn(x):
        log_p = log_prob_fn(x)

        # Use jax.debug.callback to check for NaN/inf outside of traced code
        # This allows us to raise errors/warnings during execution
        def check_log_prob(log_p_val, x_val):
            # Check for NaN - this indicates a bug in the user's log_prob function
            if jnp.any(jnp.isnan(log_p_val)):
                raise ValueError(
                    """
                    Log probability returned NaN. This indicates a bug in your log_prob function.
                    Common causes: invalid mathematical operations (e.g., log of negative number),
                    division by zero, or numerical instability.
                    """
                )

            # Check for +inf - this is suspicious (infinite probability is unusual)
            if jnp.any(jnp.isposinf(log_p_val)):
                warnings.warn(
                    "Log probability returned +inf. This represents infinite probability density, "
                    "which may indicate an issue with your log_prob function. "
                    "Please verify this is intentional.",
                    UserWarning,
                    stacklevel=4
                )

        # Call the check function as a side effect
        jax.debug.callback(check_log_prob, log_p, x)

        return log_p

    return validated_fn


def _validate_grad_log_prob_output(log_prob_fn: Callable) -> Callable:
    """Wrap gradient of log_prob function with NaN/inf validation.

    This wrapper checks the gradient output for:
    - NaN values: Raises a ValueError as these indicate bugs in the log_prob function
    - inf values: Raises a ValueError as infinite gradients will cause sampling issues

    Args:
        log_prob_fn (Callable): The log probability function whose gradient to validate.

    Returns:
        Callable: Validated gradient function.

    Raises:
        ValueError: If the gradient returns NaN or inf.
    """
    grad_fn = jax.grad(log_prob_fn)

    def validated_grad_fn(x):
        grad = grad_fn(x)

        # Use jax.debug.callback to check for NaN/inf outside of traced code
        def check_grad(grad_val, x_val):
            # Check for NaN - this indicates a bug in the user's log_prob function
            if jnp.any(jnp.isnan(grad_val)):
                raise ValueError(
                    """
                    Gradient of log probability returned NaN. This indicates a bug in your log_prob function.
                    Common causes: discontinuous functions, invalid mathematical operations, or numerical instability.
                    """
                )

            # Check for inf - this will cause issues in Hamiltonian sampling
            if jnp.any(jnp.isinf(grad_val)):
                raise ValueError(
                    """
                    Gradient of log probability returned inf. This will cause sampling failures.
                    Common causes: functions with undefined derivatives (e.g., sqrt at 0, 1/x near 0),
                    or extremely steep gradients indicating numerical instability.
                    """
                )

        # Call the check function as a side effect
        jax.debug.callback(check_grad, grad, x)

        return grad

    return validated_grad_fn