"""Hamiltonian Monte Carlo kernels and integrators."""

import jax
import jax.numpy as jnp
from typing import Callable, Optional


def leapfrog(x: jnp.ndarray, p: jnp.ndarray, grad_fn: Callable, step_size: float, L: int):
    r"""Run a vectorized leapfrog integrator.

    Args:
        x (jnp.ndarray): Current position with shape ``(n_chains, dim)``.
        p (jnp.ndarray): Current momentum with shape ``(n_chains, dim)``.
        grad_fn (Callable): Gradient of the log density :math:`\nabla \log \pi(x)`
            mapping arrays of shape ``(n_chains, dim)`` to gradients of the
            same shape.
        step_size (float): Integration step size.
        L (int): Number of leapfrog steps to integrate.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Proposed position and momentum with
            shape ``(n_chains, dim)``.
    """
    # initial half step for momentum
    x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
    p      = p - 0.5 * step_size * x_grad

    # body for the first L-1 full steps
    def body(_, state):
        x, p = state
        x = x + step_size * p                        # full position step
        x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
        p = p - step_size * x_grad                   # full momentum step
        return (x, p)

    # iterate body L-1 times
    x, p = jax.lax.fori_loop(0, L - 1, body, (x, p))

    # final position update (last half step for x)
    x = x + step_size * p

    # final half step for momentum
    x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
    p = p - 0.5 * step_size * x_grad

    # flip momentum for reversibility
    return x, -p

    
def hmc(log_prob: Callable,
        initial: jnp.ndarray,
        n_samples: int,
        grad_fn : Optional[Callable] = None,
        step_size: float = 0.1,
        L: int = 10,
        n_chains: int = 1,
        n_thin: int = 1,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    """Run a Hamiltonian Monte Carlo (HMC) sampler.

    Args:
        log_prob (Callable): Log-probability function for the target
            distribution that accepts a ``(dim,)`` array and returns a scalar.
        initial (jnp.ndarray): Initial parameter vector of shape ``(dim,)``
            used for all chains.
        n_samples (int): Number of samples to draw per chain after thinning.
        grad_fn (Callable, optional): Gradient of ``log_prob``. When ``None``
            the gradient is computed using ``jax.grad``. Defaults to ``None``.
        step_size (float): Step size for the leapfrog integrator. Defaults to
            ``0.1``.
        L (int): Number of leapfrog steps to take per proposal. Defaults to
            ``10``.
        n_chains (int): Number of parallel chains to simulate. Defaults to ``1``.
        n_thin (int): Thinning interval; store every ``n_thin`` samples.
            Defaults to ``1`` (no thinning).
        key (jax.random.PRNGKey): Random number generator key. Defaults to
            ``jax.random.PRNGKey(0)``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the samples with
            shape ``(n_chains, n_samples, dim)`` and the acceptance rate for
            each chain with shape ``(n_chains,)``.

    Notes:
        The algorithm uses a leapfrog integrator followed by a
        Metropolis-Hastings correction. NaN gradients are replaced with zeros,
        and the implementation is vectorized over chains.
    """
    ### Setup
    # JIT access to functions
    log_prob_fn = jax.jit(jax.vmap(log_prob))

    if grad_fn is None:
        grad_fn = jax.jit(jax.vmap(jax.grad(log_prob))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

    # Integers
    dim = len(initial)
    total_iterations = n_samples * n_thin

    # Random Numbers
    subkey_momentum, subkey_acceptance = jax.random.split(key, 2)
    
    # Shape (total_iterations + 1, ...), the +1 is for the initial momentum
    p_rngs = jax.random.normal(subkey_momentum, shape=(total_iterations + 1, n_chains, dim))
    acceptance_rngs = jnp.log(jax.random.uniform(subkey_acceptance, shape=(total_iterations, n_chains,), minval=1e-6, maxval=1))
    
    spread: float = 0.1
    chains_init = initial[None, :] + spread * p_rngs[0] # shape (n_chains, dim)
    accepts_init = jnp.zeros(n_chains) # shape (n_chains,)
    
    def main_loop(carry, lst_i):
        x, accepts = carry
        p, log_u = lst_i

        current_x = x.copy()
        current_p = p.copy() # Shape (n_chains, dim)

        ### Leapfrog integration
        # Proposed state
        x, p = leapfrog(x, p, grad_fn, step_size, L) # Shape (n_chains, dim), (n_chains, dim)

        ### Metropolis acceptance
        current_log_probs = log_prob_fn(current_x)      # Shape (n_chains,)
        proposal_log_probs = log_prob_fn(x)             # Shape (n_chains,)
        current_K = 0.5 * jnp.sum(current_p**2, axis=1) # Shape (n_chains,)
        proposal_K = 0.5 * jnp.sum(p**2, axis=1)        # Shape (n_chains,)

        dH = (proposal_log_probs + proposal_K) - (current_log_probs + current_K)
        log_accept_prob = jnp.minimum(0.0, -dH) # Shape (n_chains,)


        # Create mask for accepted proposals. True means accept the proposal, false means reject the proposal
        accept_mask = log_u < log_accept_prob # Shape (n_chains,)
        current_x = jnp.where(accept_mask[:, None], x, current_x)
        
        accepts += accept_mask.astype(int) # shape (n_chains,)
        
        return (current_x, accepts), current_x
    
    carry, samples = jax.lax.scan(
        main_loop, 
        init=(chains_init, accepts_init), 
        xs=(p_rngs[1:], acceptance_rngs) 
    )
    accepts = carry[1]

    # Transpose samples to get the correct shape: (n_chains, n_samples, dim)
    samples = jnp.transpose(samples, (1, 0, 2))

    # Calculate acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates