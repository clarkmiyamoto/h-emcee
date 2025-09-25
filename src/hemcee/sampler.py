from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp

from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.dual_averaging import DAState, da_cond_update
from hemcee.proposal import accept_proposal


class HamiltonianEnsembleSampler:
    """
    Hamiltonian Ensemble samplers. 
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        grad_log_prob: Callable = None,
        step_size: float = 0.2,
        L: int = 10,
        move = hmc_walk_move,
        target_accept: float = 0.8, # Dual Averaging parameters
        t0: float = 10.0,
        mu: float = 0.05,
        gamma: float = 0.05,
        kappa: float = 0.75,
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

        self.move = move

        # Dual Averaging parameters
        self.target_accept = target_accept
        self.t0 = t0
        self.mu = mu
        self.gamma = gamma
        self.kappa = kappa

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
        da_state = DAState(
            step_size=self.step_size,
            H_bar=jnp.array(0.0, dtype=jnp.float_),
            log_epsilon_bar=jnp.array(jnp.log(self.step_size), dtype=jnp.float_)
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

            step_size = da_state.step_size

            #### Group 1 / Group 2 Update
            # Update group 1 using group 2 as complement
            group1_proposed, log_prob_group1 = self.move(group1, group2, 
                step_size, keys[0],
                self.log_prob, self.grad_log_prob, 
                self.L
            )
            # Update group 2 using group 1 as complement  
            group2_proposed, log_prob_group2 = self.move(group2, group1, 
                step_size, keys[1],
                self.log_prob, self.grad_log_prob, 
                self.L
            )
            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[0])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_prob_group2, keys[1])

            #### Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            current_accept_rate = jnp.mean(all_accepts)
            
            #### Dual Averaging (simplified for now - no adaptation)
            da_state = da_cond_update(iteration, warmup, 
                current_accept_rate, target_accept=self.target_accept, 
                t0=self.t0, mu=self.mu, gamma=self.gamma, kappa=self.kappa, 
            state=da_state)
            
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

class EnsembleSampler:

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        move = walk_move,
    ) -> None:
        self.total_chains = int(total_chains)
        if self.total_chains < 4:
            raise ValueError("`self.total_chains` must be at least 4 to form meaningful ensemble groups")

        self.dim = int(dim)

        self.log_prob = jax.jit(jax.vmap(log_prob))
        self.move = move

    def run_mcmc(self, 
               key: jax.random.PRNGKey,
               initial_state: jnp.ndarray, 
               num_samples: int,
               warmup: int = 1000,
               thin_by=1,
               show_progress: bool = False,
               **kwargs
               ) -> Tuple[jnp.ndarray, dict]:
        '''
        Args
        
        Args:
            key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
            initial_state: Initial state
            num_samples: Number of post-warmup samples
            warmup: Number of warmup samples.
            thin_by: Drops every `thin_by` number of samples. Default: 1 (no thinning).
            show_progress: Whether to show progress bar. 
                NOTE: THIS WILL SIGNIFICANTLY DEGRADE PERFORMANCE!
        '''
        if show_progress:
            raise NotImplementedError("`show_progress=True` is not supported yet")
        
        if thin_by < 1:
            raise ValueError("`thinning` must 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")
            
        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")
        
        total_samples = warmup + num_samples * thin_by

        # Split chains into two groups
        group1_size = self.total_chains // 2
        group2_size = self.total_chains - group1_size

        print(f"Using {self.total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Keys for RNG
        keys_per_iter = 2
        total_rng_calls = total_samples * keys_per_iter
        keys = jax.random.split(key, total_rng_calls).reshape(total_samples, keys_per_iter, 2)
        
        # Initialize diagnostics
        diagnostics = {
            'acceptance_rate': 0.0,
        }
        
        # Initialize ensemble
        group1 = initial_state[:group1_size]
        group2 = initial_state[group1_size:]
        
        def body(carry, keys):
            group1, group2, diagnostics = carry

            # Construt Proposal
            group1_proposed, log_prob_group1 = self.move(group1, group2, keys[0], self.log_prob, **kwargs)
            group2_proposed, log_prob_group2 = self.move(group2, group1, keys[1], self.log_prob, **kwargs)

            # Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[0])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_prob_group2, keys[1])

            #### Logging diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            current_accept_rate = jnp.mean(all_accepts)
            diagnostics['acceptance_rate'] += current_accept_rate
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])


            return (group1, group2, diagnostics), final_states
                
        carry, samples = jax.lax.scan(body, init=(group1, group2, diagnostics), xs=keys)
        _, _, diagnostics = carry

        # Return post-warmup samples
        post_warmup_samples = samples[warmup:] if warmup > 0 else samples

        # Thinning
        if thin_by > 1:
            post_warmup_samples = post_warmup_samples[::thin_by]

        return post_warmup_samples, diagnostics
            