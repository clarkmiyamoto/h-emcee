from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from .sampler import BaseSampler
from .sampler_utils import calculate_batch_size, accept_proposal, batched_scan
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.adaptation.adapter import select_adapter, Adapter
from hemcee.adaptation.dual_averaging import DAParameters
from hemcee.adaptation.chees import ChEESParameters
from hemcee.backend.backend import Backend

class HamiltonianEnsembleSampler(BaseSampler):
    """Hamiltonian ensemble sampler with optional dual averaging and ChEES adaptation.

    Attributes:
        total_chains (int): Total number of ensemble chains.
        dim (int): Dimensionality of the target distribution.
        log_prob (Callable): Vectorized log-probability function.
        grad_log_prob (Callable): Vectorized gradient of the log probability.
        step_size (float): Leapfrog step size.
        L (int): Number of leapfrog steps per move.
        move (Callable): Proposal function updating each ensemble group.
        adapter (Adapter): Adapter for step size and integration time adaptation.
        adapter_state: Current state of the adapter.
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        step_size: float = 0.1,
        L: int = 10,
        move = hmc_walk_move,
        backend: Backend = None,
        adapt_step_size: bool | DAParameters = True,
        adapt_length: bool | ChEESParameters = True,
    ) -> None:
        """Initialise the sampler configuration.

        Args:
            total_chains (int): Total number of chains in the ensemble.
            dim (int): Dimensionality of the target distribution.
            log_prob (Callable): Callable returning the log density of the
                target distribution (doesn't need to be normalized).
            step_size (float): Leapfrog step size. Defaults to ``0.2``.
            L (int): Number of leapfrog steps per proposal. Defaults to ``10``.
            move (Callable): Proposal function used for ensemble updates.
            backend: Backend for storing chain data.
        """
        # Initialize base class
        super().__init__(total_chains, dim, log_prob, move, backend)
        
        # Hamiltonian-specific parameters
        self.step_size = step_size
        self.L = L

        # Dual Averaging / ChEES Adapter
        self.adapter = select_adapter(adapt_step_size, adapt_length, self.move, self.step_size, self.L)
        self.adapter_state = self.adapter.init(self.dim)

    def run_mcmc(self,
                 key: jax.random.PRNGKey,
                 initial_state: jnp.ndarray,
                 num_samples: int,
                 warmup: int = 1000,
                 thin_by=1,
                 batch_size: int = None,
                 show_progress: bool = False,
                 ) -> Tuple[jnp.ndarray, dict]:
        """Run the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            initial_state (jnp.ndarray): Initial ensemble state with shape
                ``(total_chains, dim)``.
            num_samples (int): Number of post-warmup samples to retain.
            warmup (int): Number of warmup iterations. Defaults to ``1000``.
            thin_by (int): Keep every ``thin_by``
                sample. Defaults to ``1`` (no thinning).
            show_progress (bool): Whether to display a progress bar. Defaults
                to ``False``.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and dual averaging state.
        """
        ########################################################
        # Check inputs for sanity
        ########################################################                
        self._validate_mcmc_inputs(warmup, num_samples, thin_by, initial_state)
        
        ########################################################
        # MCMC Code
        ########################################################
        # Split chains into two groups
        group1_size = self.total_chains // 2
        group2_size = self.total_chains - group1_size
        
        print(f"Using {self.total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Initialize ensemble
        group1 = initial_state[:group1_size]
        group2 = initial_state[group1_size:]

        # Calculate batch sizes
        warmup_batch_size = calculate_batch_size(self.total_chains, self.dim, warmup, batch_size)
        main_batch_size = calculate_batch_size(self.total_chains, self.dim, num_samples * thin_by, batch_size)
        
        # Warmup
        # Contains dual averaging + ChEES updating
        if warmup > 0:
            print('Starting warmup...')
            group1, group2 = self._mcmc_warmup(key, group1, group2, warmup, 
                                               self.adapter, warmup_batch_size, 
                                               show_progress)
            print('Warmup complete.')

        # Main sampling
        # Statically sets step size & integration length from warmup
        print('Starting main sampling...')
        if warmup > 0:
            # Extract final adapted values
            step_size, L = self.adapter.finalize(self.adapter_state)
        else:
            step_size = self.step_size
            L = self.L
        self._mcmc_main(key, group1, group2, num_samples, thin_by, step_size, L, main_batch_size, show_progress, warmup_offset=warmup)
        print('Main sampling complete.')

        return self.get_chain(discard=warmup, thin=thin_by)
    
    def _mcmc_warmup(self,
               key: jax.random.PRNGKey,
               group1: jnp.ndarray,
               group2: jnp.ndarray,
               warmup: int,
               adapter: Adapter,
               batch_size: int,
               show_progress: bool,
        ):
        """Run the warmup phase of the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Initial ensemble state for group 1 with shape
                ``(n_chains_per_group, dim)``.
            group2 (jnp.ndarray): Initial ensemble state for group 2 with shape
                ``(n_chains_per_group, dim)``.
            adapter (Adapter): Adapter for step size and integration time adaptation.
            warmup (int): Number of warmup iterations.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and adapter state.
        """
        def body(carry, keys):
            group1, group2, adapter_state, diagnostics = carry

            #### Group 1 / Group 2 Update
            step_size, L = adapter.value(adapter_state)
            group1_proposed, log_accept_prob_1, momentum_projected_1, proposed_log_prob_1 = self.move(group1, group2, 
                step_size, keys[0],
                self.log_prob, self.grad_log_prob, 
                L
            )
            adapter_state_after_group1 = adapter.update(
                state=adapter_state, 
                log_accept_rate=log_accept_prob_1, 
                position_current = group1,
                position_proposed = group1_proposed,
                momentum_proposed = momentum_projected_1,
                group2=group2,
                integration_time_jittered = step_size * L)
            
            # Update group 2 using group 1 as complement
            step_size, L = adapter.value(adapter_state_after_group1)  
            group2_proposed, log_accept_prob_2, momentum_projected_2, proposed_log_prob_2 = self.move(group2, group1, 
                step_size, keys[1],
                self.log_prob, self.grad_log_prob, 
                L
            )
            adapter_state_after_group2 = adapter.update(
                state=adapter_state_after_group1, 
                log_accept_rate=log_accept_prob_2, 
                position_current = group2,
                position_proposed = group2_proposed,
                momentum_proposed = momentum_projected_2,
                group2=group1,
                integration_time_jittered = step_size * L)
            
            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_accept_prob_1, keys[2])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_accept_prob_2, keys[3])

            #### Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            final_states = jnp.concatenate([group1, group2])
            
            #### Construct return state
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            final_log_probs = self.log_prob(final_states)
            
            return (group1, group2, adapter_state_after_group2, new_diagnostics), (final_states, final_log_probs, all_accepts)
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        
        n_keys = 4
        keys = jax.random.split(key, n_keys * warmup).reshape(warmup, n_keys, 2)
        
        carry, self.backend = batched_scan(body, 
                             init_carry=(group1, group2, self.adapter_state, diagnostics), 
                             xs=keys, 
                             batch_size=batch_size, 
                             backend=self.backend,
                             show_progress=show_progress,
                             offset=0)
        group1, group2, adapter_state_final, diagnostics = carry

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / warmup

        self.diagnostics_warmup = diagnostics
        self.adapter_state = adapter_state_final
        
        # Mark end of warmup phase
        self.backend.mark_warmup_end()

        return group1, group2

    def _mcmc_main(self,
                   key: jax.random.PRNGKey,
                   group1: jnp.ndarray,
                   group2: jnp.ndarray,
                   num_samples: int,
                   thin_by: int,
                   step_size: float,
                   L: int,
                   batch_size: int,
                   show_progress: bool,
                   warmup_offset: int = 0,
        ):
        """Run the main sampling phase of the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Ensemble state for group 1 with shape
                ``(n_chains_per_group, dim)``.
            group2 (jnp.ndarray): Ensemble state for group 2 with shape
                ``(n_chains_per_group, dim)``.
            num_samples (int): Number of post-warmup samples to retain.
            thin_by (int): Keep every ``thin_by`` sample.
            step_size (float): Step size for the leapfrog integrator.
            L (int): Number of leapfrog steps per proposal.
            warmup_offset (int): Number of warmup iterations (for backend indexing).

        Returns:
            samples (jnp.ndarray): Post-warmup samples with shape
                ``(num_samples, total_chains, dim)``.
        """

        ########################################################
        # Sampling Loop Body
        ########################################################
        def body(carry, keys):
            group1, group2, diagnostics = carry

            #### Group 1 / Group 2 Update
            # Update group 1 using group 2 as complement
            group1_proposed, log_accept_prob_1, momentum_projected_1, proposed_log_prob_1 = self.move(group1, group2, 
                step_size, keys[0],
                self.log_prob, self.grad_log_prob, 
                L
            )
            # Update group 2 using group 1 as complement  
            group2_proposed, log_accept_prob_2, momentum_projected_2, proposed_log_prob_2 = self.move(group2, group1, 
                step_size, keys[1],
                self.log_prob, self.grad_log_prob, 
                L
            )
            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_accept_prob_1, keys[2])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_accept_prob_2, keys[3])

            #### Diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])
            final_log_probs = self.log_prob(final_states)
            
            return (group1, group2, new_diagnostics), (final_states, final_log_probs, all_accepts)
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        n_keys_per_iter = 4
        total_samples = num_samples * thin_by
        keys = jax.random.split(key, n_keys_per_iter * total_samples).reshape(total_samples, n_keys_per_iter, 2)

        carry, self.backend = batched_scan(body, 
                             init_carry=(group1, group2, diagnostics), 
                             xs=keys, 
                             batch_size=batch_size, 
                             backend=self.backend,
                             show_progress=show_progress,
                             offset=warmup_offset)
        group1, group2, diagnostics = carry

        # Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics


            
