import jax
import jax.numpy as jnp
from typing import Callable

def leapfrog_side_move(q1, 
                       p1_current, 
                       grad_fn, 
                       beta_eps, 
                       L,
                       n_chains_per_group,
                       diff_particles_group2):
    '''
    Args
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1: momentum of first group of chains. Shape (n_chains_per_group,)
        - grad_fn: gradient of the potential function. Function of shape (n_chains, dim) -> (n_chains, dim)
        - beta_eps: half of the step size. Scalar
        - L: number of leapfrog steps. Scalar
        - L: number of leapfrog steps
    
    Returns
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1_current: momentum of first group of chains. Shape (n_chains_per_group,)
    '''
    # Initial half-step for momentum - VECTORIZED
    grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)
    
    # Compute dot products between gradients and difference particles - VECTORIZED
    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
    p1_current -= 0.5 * beta_eps * gradient_projections # Shape (n_chains_per_group,)

    # Full leapfrog steps
    for step in range(L):
        q1 += beta_eps * (jnp.expand_dims(p1_current, axis=1) * diff_particles_group2) # Shape: (n_chains_per_group, dim)
        
        if (step < L - 1):
            grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
            grad1 = jnp.nan_to_num(grad1, nan=0.0)

            gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
            p1_current -= beta_eps * gradient_projections # Shape (n_chains_per_group,)
    
    # Final half-step for momentum - VECTORIZED 
    grad1 = grad_fn(q1)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)

    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
    p1_current -= 0.5 * beta_eps * gradient_projections

    return q1, p1_current # Shape (n_chains_per_group, dim), (n_chains_per_group,)

def hamiltonian_side_move(potential_func: Callable, 
                          initial: jnp.ndarray, 
                          n_samples: int,
                          grad_fn: Callable = None, 
                          n_chains_per_group: int = 5,
                          step_size: float = 0.01, 
                          L: int = 10, 
                          beta: float = 1.0,
                          n_thin = 1,
                          key=jax.random.PRNGKey(0)):
    """Hamiltonian Side Move (HSM) sampler implementation using JAX.
    
    This function implements the HSM algorithm, which uses Hamiltonian dynamics with side moves
    to propose new states in the Markov chain. It maintains two groups of chains that interact
    with each other through Hamiltonian dynamics, allowing for better exploration of the target
    distribution. It supports multiple chains per group and thinning of samples.
    
    Args:
        potential_func: Function that computes the potential energy (negative log probability)
                       of the target distribution. Should take a single argument (parameters)
                       and return a scalar.
        initial: Initial parameter values for the Markov chains. Can be any shape, will be
                flattened internally.
        n_samples: Number of samples to generate per chain
        n_chains_per_group: Number of chains in each of the two groups. Total number of chains
                           will be 2 * n_chains_per_group. Default: 5
        step_size: Step size for the leapfrog integrator. Controls the discretization of
                Hamiltonian dynamics. Default: 0.01
        L: Number of leapfrog steps per proposal. Controls how far each proposal
                   can move. Default: 10
        beta: Temperature parameter that controls the strength of the Hamiltonian dynamics.
              Higher values lead to more aggressive exploration. Default: 1.0
        n_thin: Thinning interval for the samples. Only every nth sample is stored.
                Default: 1 (no thinning)
        key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
    
    Returns:
        tuple: A tuple containing:
            - samples: Array of shape (2*n_chains_per_group, n_samples, *initial.shape)
                      containing the MCMC samples
            - acceptance_rates: Array of shape (2*n_chains_per_group,) containing the
                              acceptance rate for each chain
    
    Notes:
        - The algorithm uses two groups of chains that interact through Hamiltonian dynamics
        - Each chain in one group interacts with two randomly selected chains from the other group
        - The interaction is mediated through the difference between the selected chains
        - Metropolis-Hastings acceptance is used to ensure detailed balance
        - NaN gradients are handled by replacing them with zeros
        - The implementation is vectorized to run multiple chains in parallel
    """

    ### ERROR CHECKING
    if (n_chains_per_group <= 1):
        raise ValueError("n_chains_per_group must be greater than 1")
    
    # Initialize
    dim = len(initial)
    total_chains = 2 * n_chains_per_group

    potential_func_vmap = jax.jit(jax.vmap(potential_func))           # F: (n_chains, dim) -> (n_chains,)
    
    if grad_fn is None:
        grad_fn_vmap = jax.jit(jax.vmap(jax.grad(potential_func))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn_vmap = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

    # Create initial states with small random perturbations

    states = jnp.tile(initial.flatten(), (total_chains, 1)) + 0.1 * jax.random.normal(key, shape=(total_chains, dim)) # Shape (total_chains, dim)
    
    # Split into two groups
    group1_indices = jnp.arange(n_chains_per_group)               # Shape (n_chains_per_group,)
    group2_indices = jnp.arange(n_chains_per_group, total_chains) # Shape (n_chains_per_group,)

    states_group1 = states[:n_chains_per_group] # Shape (n_chains_per_group, dim)
    states_group2 = states[n_chains_per_group:] # Shape (n_chains_per_group, dim)

    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin

    # Storage for samples and acceptance tracking
    accepts = jnp.zeros(total_chains)

    # Precompute some constants for efficiency
    beta_eps = beta * step_size

    keys_per_iter = 6
    all_keys = jax.random.split(key, total_iterations * keys_per_iter).reshape(total_iterations, keys_per_iter, 2)
    
    
    def main_loop(carry, keys):
        #---------------------------------------------
        # Unpack input
        #---------------------------------------------
        states, accepts = carry # Array shapped (total_chains, dim), integer
        
        # Store current state from all chains (only every n_thin iterations)
        # if (i % n_thin == 0) and (sample_idx < n_samples):
        #     samples = samples.at[:, sample_idx, :].set(states)
        #     sample_idx += 1

        #---------------------------------------------
        # First group update - VECTORIZED
        #---------------------------------------------

        # For each particle in group 1, randomly select TWO particles from group 2
        keys_choices= jax.random.split(keys[0], n_chains_per_group)
        choices = jax.vmap(lambda k: jax.random.choice(k, group2_indices, shape=(2,), replace=False))(keys_choices) # Shape (n_chains_per_group, 2)
        random_indices1_from_group2 = choices[:, 0]
        random_indices2_from_group2 = choices[:, 1]

        # Get the two sets of selected particles from group 2. Shape: (n_chains_per_group, dim)
        selected_particles1_group2 = states[random_indices1_from_group2] # Shape (n_chains_per_group, dim)
        selected_particles2_group2 = states[random_indices2_from_group2] # Shape (n_chains_per_group, dim)
        
        # Use the difference between the two particles. Shape (n_chains_per_group, dim)
        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / jnp.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p1 = jax.random.normal(keys[1], shape=(n_chains_per_group,))

        # Store current state and energy
        current_q1 = states[group1_indices].copy() # Shape (n_chains_per_group, dim)
        current_U1 = potential_func_vmap(current_q1) # Shape (n_chains_per_group, dim) -> (n_chains_per_group,)

        current_K1 = 0.5 * p1**2

        q1, p1_current = leapfrog_side_move(
            current_q1, 
            p1, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            n_chains_per_group,
            diff_particles_group2
        ) # Shape (n_chains_per_group, dim), (n_chains_per_group,)
        
        # Compute proposed energy
        proposed_U1 = potential_func_vmap(q1) # Shape (n_chains_per_group,)
        proposed_K1 = 0.5 * p1_current**2 # Shape (n_chains_per_group,)
        
        # Metropolis acceptance in log scale for numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1) # Shape (n_chains_per_group,)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob1 = jnp.minimum(0.0, -dH1) # Shape (n_chains_per_group,)
        
        # Generate log-uniform random numbers
        log_u1 = jnp.log(jax.random.uniform(keys[2], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts1 = log_u1 < log_accept_prob1 # Shape (n_chains_per_group,)
        
        # Update states - VECTORIZED
        updated_group1_states = jnp.where(accepts1[:, None], q1, states[group1_indices])
        states = states.at[group1_indices].set(updated_group1_states)
        
        # Track acceptance for first chains
        accepts += jnp.sum(jnp.count_nonzero(accepts1))
        #---------------------------------------------
        # Second group update - VECTORIZED similarly
        #---------------------------------------------
        
        # For each particle in group 2, randomly select TWO particles from group 1
        keys_choices= jax.random.split(keys[3], n_chains_per_group)
        choices = jax.vmap(lambda k: jax.random.choice(k, group1_indices, shape=(2,), replace=False))(keys_choices) # Shape (n_chains_per_group, 2)
        random_indices1_from_group1 = choices[:, 0]
        random_indices2_from_group1 = choices[:, 1]

        # Get the two sets of selected particles from group 1. Shape (n_chains_per_group, dim)
        selected_particles1_group1 = states[random_indices1_from_group1]
        selected_particles2_group1 = states[random_indices2_from_group1]

        # Use the difference between the two particles (shape: n_chains_per_group x dim)
        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / jnp.sqrt(2*n_chains_per_group)

        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p2 = jax.random.normal(keys[4], shape=(n_chains_per_group,))   
        
        # Store current state and energy
        current_q2 = states[group2_indices].copy()
        current_U2 = potential_func_vmap(current_q2)
        current_K2 = 0.5 * p2**2

        # Leapfrog integration with preconditioning
        # q2 = current_q2.copy()
        # p2_current = p2.copy()

        q2, p2_current = leapfrog_side_move(
            current_q2, 
            p2, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            n_chains_per_group, 
            diff_particles_group1
        )

        # Compute proposed energy
        proposed_U2 = potential_func_vmap(q2)
        proposed_K2 = 0.5 * p2_current**2

        # Metropolis acceptance in log scale for numerical stability
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob2 = jnp.minimum(0.0, -dH2)
        
        # Generate log-uniform random numbers
        log_u2 = jnp.log(jax.random.uniform(keys[5], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts2 = log_u2 < log_accept_prob2

        # Update states - VECTORIZED
        updated_group2_states = jnp.where(accepts2[:, None], q2, states[group2_indices])
        states = states.at[group2_indices].set(updated_group2_states)

        # Track acceptance for second chains
        accepts += jnp.sum(jnp.count_nonzero(accepts2))

        return (states, accepts), states

    carry, previous_states = jax.lax.scan(main_loop, init=(states, 0), xs=all_keys)
    current_states, accepts = carry
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / (total_iterations * n_chains_per_group * 2)
    
    return previous_states, acceptance_rates

