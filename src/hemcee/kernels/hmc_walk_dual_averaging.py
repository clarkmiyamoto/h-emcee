import jax
import jax.numpy as jnp
from typing import Callable

from hemcee.dual_averaging import DualAveraging

def leapfrog_walk_move(q: jnp.ndarray,
                       p: jnp.ndarray,
                       grad_fn: Callable,
                       beta_eps: float,
                       L: int,
                       centered: jnp.ndarray):
    """Perform a leapfrog step for the dual-averaging walk move.

    Args:
        q (jnp.ndarray): Current positions with shape
            ``(n_chains_per_group, dim)``.
        p (jnp.ndarray): Current ensemble momenta with shape
            ``(n_chains_per_group, n_chains_per_group)``.
        grad_fn (Callable): Vectorized gradient mapping ``(batch_size, dim)``
            arrays to the same shape.
        beta_eps (float): Product of the coupling constant and current step
            size.
        L (int): Number of leapfrog substeps.
        centered (jnp.ndarray): Centered complement ensemble with shape
            ``(n_chains_per_group, dim)``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Proposed positions and momenta with
            the same shapes as ``q`` and ``p``.
    """
    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0) 

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T) # Shape (n_chains_per_group, n_chains_per_group)
   

    for step in range(L):
        q += beta_eps * jnp.dot(p, centered) # Shape (n_chains_per_group, dim)

        if (step < L - 1):
            grad = grad_fn(q) # Shape (n_chains_per_group, dim)
            grad = jnp.nan_to_num(grad, nan=0.0)

            p -= beta_eps * jnp.dot(grad, centered.T)

    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0)

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

    return q, p

def hamiltonian_walk_move_dual_averaging(potential_func: Callable,
                          initial: jnp.ndarray,
                          n_samples: int,
                          grad_fn: Callable = None,
                          n_chains_per_group: int = 5,
                          epsilon: float = 0.1,
                          L: int = 10,
                          beta: float = 1.0,
                          n_thin=1,
                          n_warmup: int = 1000,
                          target_accept: float = 0.65,
                          gamma: float = 0.05,
                          t0: float = 10.0,
                          kappa: float = 0.75,
                          key=jax.random.PRNGKey(0)):
    """Run the dual-averaging Hamiltonian Walk Move sampler.

    Args:
        potential_func (Callable): Potential energy (negative log-density)
            function returning scalars for ``(dim,)`` inputs.
        initial (jnp.ndarray): Initial parameters broadcast to all chains.
        n_samples (int): Number of samples to collect per chain.
        grad_fn (Callable, optional): Gradient of ``potential_func``. Uses
            automatic differentiation when ``None``.
        n_chains_per_group (int): Number of chains per ensemble group.
            Defaults to ``5``.
        epsilon (float): Initial step size for the leapfrog integrator.
        L (int): Number of leapfrog substeps per proposal. Defaults to ``10``.
        beta (float): Interaction strength between ensembles. Defaults to
            ``1.0``.
        n_thin (int): Thinning interval; keep every ``n_thin`` sample.
            Defaults to ``1``.
        n_warmup (int): Number of warmup iterations for dual averaging.
            Defaults to ``1000``.
        target_accept (float): Target acceptance rate for dual averaging.
            Defaults to ``0.65``.
        gamma (float): Dual averaging parameter controlling shrinkage.
            Defaults to ``0.05``.
        t0 (float): Dual averaging parameter controlling initial stability.
            Defaults to ``10.0``.
        kappa (float): Dual averaging parameter controlling adaptation speed.
            Defaults to ``0.75``.
        key (jax.random.PRNGKey): Random number generator key. Defaults to
            ``jax.random.PRNGKey(0)``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, float]: Tuple containing the states
            with shape ``(total_iterations, 2 * n_chains_per_group, dim)``,
            acceptance rates with shape ``(2 * n_chains_per_group,)``, and the
            final adapted step size where
            ``total_iterations = n_samples * n_thin``.
    """
    # JIT
    potential_func_vmap = jax.jit(jax.vmap(potential_func))           # F: (n_chains, dim) -> (n_chains,)
    if grad_fn is None:
        grad_fn_vmap = jax.jit(jax.vmap(jax.grad(potential_func))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn_vmap = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

    # Sizes
    dim = len(initial)
    total_chains = 2 * n_chains_per_group
    total_iterations = n_samples * n_thin

    # Initalize States
    spread = 0.1
    states = jnp.tile(initial.flatten(), (total_chains, 1)) + spread * jax.random.normal(key, shape=(total_chains, dim)) # Shape (total_chains, dim)
    states_group1 = states[:n_chains_per_group] # Shape (n_chains_per_group, dim)
    states_group2 = states[n_chains_per_group:] # Shape (n_chains_per_group, dim)

    accepts_group1 = jnp.zeros(n_chains_per_group)
    accepts_group2 = jnp.zeros(n_chains_per_group)

    keys_per_iter = 4
    all_keys = jax.random.split(key, total_iterations * keys_per_iter).reshape(total_iterations, keys_per_iter, 2)

    beta_eps = beta * epsilon

    # Initialize Dual Averaging
    dual_averaging = DualAveraging(jnp.log(epsilon), target_accept, gamma, t0, kappa)
    
    # Initialize dual averaging state variables
    H_bar = dual_averaging.H_bar
    log_epsilon = dual_averaging.log_epsilon
    log_epsilon_bar = dual_averaging.log_epsilon_bar
    m = dual_averaging.m

    def main_loop(carry, keys):
        #---------------------------------------------
        # Unpack input
        #---------------------------------------------
        states_group1, states_group2, accepts_group1, accepts_group2, H_bar, log_epsilon, log_epsilon_bar, m = carry

        q1 = states_group1
        q2 = states_group2
        q1_current = q1.copy()
        q2_current = q2.copy()

        # Use current step size for this iteration
        current_eps = jnp.exp(log_epsilon)
        beta_eps = beta * current_eps

        ########################################################
        # Group 1
        ########################################################

        centered2 = (q2 - jnp.mean(q2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)

        # Random Momentum
        p1_current = jax.random.normal(keys[0], shape=(n_chains_per_group, n_chains_per_group))

        # Current Energy
        current_U1 = potential_func_vmap(q1_current) # Shape (n_chains_per_group,)
        current_K1 = 0.5 * jnp.sum(p1_current**2, axis=1) # Shape (n_chains_per_group,)

        # Leapfrog Integration
        q1_proposed, p1_proposed = leapfrog_walk_move(
            q1_current, 
            p1_current, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            centered2
        )
        proposed_U1 = potential_func_vmap(q1_proposed)
        proposed_K1 = 0.5 * jnp.sum(p1_proposed**2, axis=1) # Shape (n_chains_per_group,)

        # Metropolis Step in log scale for numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob1 = jnp.minimum(0.0, -dH1)
        
        # Generate log-uniform random numbers
        log_u1 = jnp.log(jax.random.uniform(keys[1], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts1 = log_u1 < log_accept_prob1

        # Log Changes
        accepts_group1 += accepts1.astype(int)
        states_group1 = jnp.where(accepts1[:, None], q1_proposed, states_group1)


        ########################################################
        # Group 2
        ########################################################

        centered1 = (q1 - jnp.mean(q1, axis=0)) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)

        # Random Momentum
        p2_current = jax.random.normal(keys[2], shape=(n_chains_per_group, n_chains_per_group))

        # Current Energy
        current_U2 = potential_func_vmap(q2_current)
        current_K2 = 0.5 * jnp.sum(p2_current**2, axis=1) # Shape (n_chains_per_group,)

        # Leapfrog Integration
        q2_proposed, p2_proposed = leapfrog_walk_move(
            q2_current, 
            p2_current, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            centered1
        )
        proposed_U2 = potential_func_vmap(q2_proposed)
        proposed_K2 = 0.5 * jnp.sum(p2_proposed**2, axis=1) # Shape (n_chains_per_group,)

        # Metropolis Step in log scale for numerical stability
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob2 = jnp.minimum(0.0, -dH2)
        
        # Generate log-uniform random numbers
        log_u2 = jnp.log(jax.random.uniform(keys[3], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts2 = log_u2 < log_accept_prob2

        # Log Changes
        accepts_group2 += accepts2.astype(int)
        states_group2 = jnp.where(accepts2[:, None], q2_proposed, states_group2)

        ########################################################
        # Dual Averaging Step
        ########################################################
        current_accept_rate = (jnp.sum(accepts1) + jnp.sum(accepts2)) / total_chains
        
        # Update dual averaging state during warmup
        def update_dual_averaging():
            new_H_bar, new_log_epsilon, new_log_epsilon_bar = dual_averaging.update(current_accept_rate)
            return new_H_bar, new_log_epsilon, new_log_epsilon_bar
        
        def no_update():
            return H_bar, log_epsilon, log_epsilon_bar
        
        H_bar_new, log_epsilon_new, log_epsilon_bar_new = jax.lax.cond(
            m < n_warmup, 
            update_dual_averaging, 
            no_update
        )
        
        m_new = m + 1

        ########################################################
        # Return
        ########################################################

        final_states = jnp.concatenate([states_group1, states_group2]) # Shape (total_chains, dim)

        return (states_group1, states_group2, accepts_group1, accepts_group2, H_bar_new, log_epsilon_new, log_epsilon_bar_new, m_new), final_states
    
    carry, previous_states = jax.lax.scan(main_loop, init=(states_group1, states_group2, accepts_group1, accepts_group2, H_bar, log_epsilon, log_epsilon_bar, m), xs=all_keys)
    states_group1, states_group2, accepts_group1, accepts_group2, H_bar_final, log_epsilon_final, log_epsilon_bar_final, m_final = carry

    # Compute acceptance rates for all chains
    acceptance_rates_group1 = accepts_group1 / (total_iterations) # Shape (n_chains_per_group,)
    acceptance_rates_group2 = accepts_group2 / (total_iterations) # Shape (n_chains_per_group,)

    acceptance_rates = jnp.concatenate([acceptance_rates_group1, acceptance_rates_group2]) # Shape (total_chains,)

    # Final adapted step size
    final_epsilon = jnp.exp(log_epsilon_bar_final)

    return previous_states, acceptance_rates, final_epsilon