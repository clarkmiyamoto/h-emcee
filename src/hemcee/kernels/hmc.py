import jax 
import jax.numpy as jnp
from typing import Callable, Optional


def leapfrog(x: jnp.ndarray, p: jnp.ndarray, grad_fn: callable, step_size: float, L: int):
    """
    Vectorised leap-frog integrator without Python-side conditionals.

    Args:
        - x: current position. Shape (n_chains, dim)
        - p: current momentum. Shape (n_chains, dim)
        - grad_fn: callable - ∇log π(x). Function of shape (n_chains, dim) -> (n_chains, dim)
        - step_size: Step size.
        - L: Number of leap-frog steps

    Returns:
        - x: proposed position. Shape (n_chains, dim)
        - p: proposed momentum. Shape (n_chains, dim)
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
    """Hamiltonian Monte Carlo (HMC) sampler implementation using JAX.
    
    This function implements the HMC algorithm, which uses Hamiltonian dynamics to propose
    new states in the Markov chain. It supports multiple chains and thinning of samples.
    
    Args:
        log_prob: Function that computes the log probability density of the target distribution.
                 Should take a single argument (parameters) (shaped (dim,)) and return a scalar.
        initial: Initial parameter values for the Markov chains. Shape (dim,)
        n_samples: Number of samples to generate per chain
        step_size: Step size for the leapfrog integrator. Controls the discretization of
                Hamiltonian dynamics. Default: 0.1
        L: Number of leapfrog steps per proposal. Controls how far each proposal can move.
           Default: 10
        n_chains: Number of independent Markov chains to run in parallel. Default: 1
        n_thin: Thinning interval for the samples. Only every nth sample is stored.
                Default: 1 (no thinning)
        key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
    
    Returns:
        tuple: A tuple containing:
            - samples: Array of shape (n_chains, n_samples, dim) containing the MCMC samples
            - acceptance_rates: Array of shape (n_chains,) containing the acceptance rate
                              for each chain
    
    Notes:
        - The algorithm uses the leapfrog integrator to simulate Hamiltonian dynamics
        - Metropolis-Hastings acceptance is used to ensure detailed balance
        - NaN gradients are handled by replacing them with zeros
        - The implementation is vectorized to run multiple chains in parallel
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