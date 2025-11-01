
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .sampler import BaseSampler
from .sampler_utils import calculate_batch_size, accept_proposal, batched_scan
from hemcee.adaptation.nuts import select_nuts_step
from hemcee.adaptation.adapter import select_adapter
from hemcee.adaptation.dual_averaging import DAParameters, DualAveragingAdapter
from hemcee.backend.backend import Backend

from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move

class EnsembleNUTS(BaseSampler):
    
    def __init__(self, 
                 total_chains: int, 
                 dim: int, 
                 log_prob: Callable, 
                 move: Callable = hmc_walk_move, 
                 step_size: float = 0.1,
                 max_tree_depth: int = 5,
                 backend: Backend = None,
                 adapt_step_size: bool | DAParameters = True,):
        # Configure NUTS step function based on move type
        nuts_move = select_nuts_step(move)
        super().__init__(total_chains, dim, log_prob, nuts_move, backend)
        
        # NUTS-specific parameters
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        
        # Adapter for step size
        self.adapter = select_adapter(adapt_step_size, False, step_size, None)
        self.adapter_state = self.adapter.init(self.dim)
    
    def run_mcmc(self,
                 key: jax.random.PRNGKey,
                 initial_state: jnp.ndarray,
                 num_samples: int,
                 warmup: int = 1000,
                 thin_by: int = 1,
                 batch_size: int = None,
                 show_progress: bool = False,
                 ) -> Tuple[jnp.ndarray, dict]:
        """Run the NUTS ensemble sampler.
        
        Args:
            key: Random number generator key
            initial_state: Initial ensemble state with shape (total_chains, dim)
            num_samples: Number of post-warmup samples to retain
            warmup: Number of warmup iterations
            thin_by: Keep every thin_by sample
            batch_size: Batch size for processing
            show_progress: Whether to display progress bar
            
        Returns:
            Tuple of (samples, diagnostics)
        """
        self._validate_mcmc_inputs(warmup, num_samples, thin_by, initial_state)
        warmup_batch_size = calculate_batch_size(self.total_chains, self.dim, warmup, batch_size)
        main_batch_size = calculate_batch_size(self.total_chains, self.dim, num_samples * thin_by, batch_size)

        group1_size = self.total_chains // 2
        group2_size = self.total_chains - group1_size
        group1 = initial_state[:group1_size]
        group2 = initial_state[group1_size:]

        print(f"Using {self.total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")

        if warmup > 0:
            print('Starting warmup...')
            group1, group2 = self._mcmc_warmup(key, group1, group2, warmup, 
                                               self.adapter, warmup_batch_size, 
                                               show_progress)
            step_size, _ = self.adapter.finalize(self.adapter_state)
            print('Warmup complete.')
        else:
            step_size = self.step_size
        
        print('Starting main sampling...')
        self._mcmc_main(key, group1, group2, num_samples, thin_by, step_size, main_batch_size, show_progress, warmup_offset=warmup)
        print('Main sampling complete.')
        
        # Return samples and diagnostics
        self.diagnostics = {
            'warmup': self.diagnostics_warmup,
            'main': self.diagnostics_main
        }
        
        return self.get_chain(discard=warmup, thin=thin_by)
    
    def _mcmc_warmup(self,
                     key: jax.random.PRNGKey,
                     group1: jnp.ndarray,
                     group2: jnp.ndarray,
                     warmup: int,
                     adapter,
                     batch_size: int,
                     show_progress: bool):
        """Run warmup phase with step size adaptation."""
        def body(carry, keys):
            group1, group2, step_size, adapter_state, diagnostics = carry
            
            # Update group 1 using group 2 as complement
            group1_proposed, log_accept_prob1 = self.move(
                keys[0], group1, group2, self.log_prob, self.grad_log_prob,
                step_size, self.max_tree_depth
            )
            
            # Update group 2 using group 1 as complement
            group2_proposed, log_accept_prob2 = self.move(
                keys[1], group2, group1, self.log_prob, self.grad_log_prob,
                step_size, self.max_tree_depth
            )
            
            # Accept/reject proposals (NUTS already handles MH internally via log_accept_prob)
            group1, accept1 = accept_proposal(group1, group1_proposed, log_accept_prob1, keys[2])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_accept_prob2, keys[3])
            
            # Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            current_accept_rate = jnp.mean(all_accepts)
            final_states = jnp.concatenate([group1, group2])
            final_log_probs = self.log_prob(final_states)
            
            # Adaptation
            adapter_state_new = adapter.update(adapter_state, current_accept_rate, final_states)
            step_size_new, _ = adapter.value(adapter_state_new)
            
            # Update diagnostics
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            return (group1, group2, step_size_new, adapter_state_new, new_diagnostics), \
                   (final_states, final_log_probs, all_accepts)
        
        diagnostics = {'accepts': jnp.zeros(self.total_chains)}
        n_keys = 4
        keys = jax.random.split(key, n_keys * warmup).reshape(warmup, n_keys, 2)
        
        carry, self.backend = batched_scan(
            body,
            init_carry=(group1, group2, self.step_size, self.adapter_state, diagnostics),
            xs=keys,
            batch_size=batch_size,
            backend=self.backend,
            show_progress=show_progress,
            offset=0
        )
        
        group1, group2, step_size_final, adapter_state_final, diagnostics = carry
        
        # Update adapter state and diagnostics
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / warmup
        self.diagnostics_warmup = diagnostics
        self.adapter_state = adapter_state_final
        
        # Mark end of warmup
        self.backend.mark_warmup_end()
        
        return group1, group2
    
    def _mcmc_main(self,
                   key: jax.random.PRNGKey,
                   group1: jnp.ndarray,
                   group2: jnp.ndarray,
                   num_samples: int,
                   thin_by: int,
                   step_size: float,
                   batch_size: int,
                   show_progress: bool,
                   warmup_offset: int):
        """Run main sampling phase with fixed step size."""
        def body(carry, keys):
            group1, group2, diagnostics = carry
            
            # Update group 1 using group 2 as complement
            group1_proposed, log_accept_prob1 = self.move(
                keys[0], group1, group2, self.log_prob, self.grad_log_prob,
                step_size, self.max_tree_depth
            )
            
            # Update group 2 using group 1 as complement
            group2_proposed, log_accept_prob2 = self.move(
                keys[1], group2, group1, self.log_prob, self.grad_log_prob,
                step_size, self.max_tree_depth
            )
            
            # Accept/reject proposals
            group1, accept1 = accept_proposal(group1, group1_proposed, log_accept_prob1, keys[2])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_accept_prob2, keys[3])
            
            # Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            final_states = jnp.concatenate([group1, group2])
            final_log_probs = self.log_prob(final_states)
            
            # Update diagnostics
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            return (group1, group2, new_diagnostics), (final_states, final_log_probs, all_accepts)
        
        diagnostics = {'accepts': jnp.zeros(self.total_chains)}
        n_keys = 4
        total_samples = num_samples * thin_by
        keys = jax.random.split(key, n_keys * total_samples).reshape(total_samples, n_keys, 2)
        
        carry, self.backend = batched_scan(
            body,
            init_carry=(group1, group2, diagnostics),
            xs=keys,
            batch_size=batch_size,
            backend=self.backend,
            show_progress=show_progress,
            offset=warmup_offset
        )
        
        group1, group2, diagnostics = carry
        
        # Update diagnostics
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics

