from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp

from hemcee.dual_averaging import DAState
from hemcee.moves.hmc_walk import hmc_walk

class HamiltonianEnsembleSampler:
    """
    For Hamiltonian Ensemble samplers. Walk + slide HMC
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        grad_log_prob: Callable = None,
        step_size: float = 0.2,
        L: int = 10,
        beta: float = 1.0,
        move = hmc_walk,
    ) -> None:
        self.total_chains = int(total_chains)
        if self.total_chains < 4:
            raise ValueError("`self.total_chains` must be at least 4 to form meaningful ensemble groups")

        self.dim = int(dim)

        self.log_prob = jax.jit(jax.vmap(log_prob))
        if grad_log_prob is None:
            self.grad_log_prob = jax.jit(jax.vmap(jax.grad(log_prob)))
        else:
            self.grad_log_prob = jax.jit(jax.vmap(grad_log_prob))
        
        self.step_size = step_size
        self.L = L
        self.beta = beta

        self.move = move

    def run_mcmc(self, 
               key: jax.random.PRNGKey,
               initial_state: jnp.ndarray, 
               num_samples: int,
               warmup: int = 1000,
               thin_by=1,
               adapt_step_size: bool = True,
               adapt_integration: bool = False,
               show_progress: bool = False,
               ) -> Tuple[jnp.ndarray, dict]:
        """
        Run Ensemble NUTS sampling.
        
        Args:
            key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
            initial_state: Initial state
            num_samples: Number of post-warmup samples
            warmup: Number of warmup samples.
            thin_by: Drops every `thin_by` number of samples. Default: 1 (no thinning).
            adapt_step_size: Whether to adapt step size using dual averaging
            adapt_integration: Whether to adapt integration using affine invariant NUTs
            show_progress: Whether to show progress bar. 
                NOTE: THIS WILL SIGNIFICANTLY DEGRADE PERFORMANCE!
        """

        if show_progress:
            raise NotImplementedError("`show_progress=True` is not supported yet")
        
        if adapt_integration:
            raise NotImplementedError("`adapt_integration=True` is not supported yet")
        
        if thin_by < 1:
            raise ValueError("`thinning` must 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")

        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")
        

        ########################################################
        # Code
        ########################################################
        total_samples = warmup + num_samples * thin_by

        # Split chains into two groups
        group1_size = self.total_chains // 2
        group2_size = self.total_chains - group1_size
        
        print(f"Using {self.total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Keys for RNG
        keys_per_iter = 4
        total_rng_calls = total_samples * keys_per_iter
        keys = jax.random.split(key, total_rng_calls).reshape(total_samples, 4, 2)

        # Initialize ensemble
        group1 = initial_state[:group1_size]
        group2 = initial_state[group1_size:]
        
        # Initialize dual averaging if needed
        H_bar = 0.0
        da_state = DAState(
            step_size=self.step_size,
            H_bar=H_bar,
            log_epsilon_bar=jnp.log(self.step_size)
        )

        
        # Initialize diagnostics
        diagnostics = {
            'acceptance_rate': 0.0,
            'step_size': self.step_size,
            'dual_averaging_state': da_state
        }
        
        ########################################################
        # Sampling Loop Body
        ########################################################
        def body(carry, keys_and_iter):
            group1, group2, da_state, diagnostics = carry
            keys, iteration = keys_and_iter

            #### Group A / Group B Update
            # Update group 1 using group 2 as complement
            group1, accept1 = self.move(group1, group2, 
                da_state, keys[0],
                self.log_prob, self.grad_log_prob, 
                self.L
            )
            
            # Update group 2 using group 1 as complement  
            group2, accept2 = self.move(group2, group1, 
                da_state, keys[1],
                self.log_prob, self.grad_log_prob, 
                self.L
            )
            
            #### Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            current_accept_rate = jnp.mean(all_accepts)
            
            #### Dual Averaging (simplified for now - no adaptation)
            # For now, we'll keep the step size fixed
            # TODO: Implement proper dual averaging with DAState
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])
            new_diagnostics = {
                'acceptance_rate': current_accept_rate,
                'step_size': da_state.step_size,
                'dual_averaging_state': da_state
            }
            
            return (group1, group2, da_state, new_diagnostics), final_states
            
            
        
        # Create iteration indices for dual averaging
        iterations = jnp.arange(total_samples)
        scan_input = (keys, iterations)
        
        carry, samples = jax.lax.scan(body, init=(group1, group2, da_state, diagnostics), xs=scan_input)
        group1, group2, da_state, diagnostics = carry
        
        # Return post-warmup samples
        post_warmup_samples = samples[warmup:] if warmup > 0 else samples

        # Thinning
        if thin_by > 1:
            post_warmup_samples = post_warmup_samples[::thin_by]
        
        return post_warmup_samples, diagnostics
