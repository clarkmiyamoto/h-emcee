from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from .sampler import BaseSampler
from .sampler_utils import calculate_batch_size, accept_proposal, batched_scan
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.backend.backend import Backend

class EnsembleSampler(BaseSampler):
    """Affine-invariant ensemble sampler wrapper."""

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        move = stretch_move,
        backend: Backend = None,
    ) -> None:
        """Initialise the vanilla ensemble sampler.

        Args:
            total_chains (int): Total number of chains in the ensemble.
            dim (int): Dimensionality of the target distribution.
            log_prob (Callable): Callable returning the log density of the
                target distribution.
            move (Callable): Proposal function used for ensemble updates.
            backend: Backend for storing chain data.
        """
        # Initialize base class
        super().__init__(total_chains, dim, log_prob, move, backend)

    def run_mcmc(self,
               key: jax.random.PRNGKey,
               initial_state: jnp.ndarray,
               num_samples: int,
               warmup: int = 1000,
               thin_by: int = 1,
               batch_size: int = None,
               show_progress: bool = False,
               **kwargs
               ) -> Tuple[jnp.ndarray, dict]:
        """Run the vanilla ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            initial_state (jnp.ndarray): Initial ensemble state with shape
                ``(total_chains, dim)``.
            num_samples (int): Number of post-warmup samples to retain.
            warmup (int): Number of warmup iterations. Defaults to ``1000``.
            thin_by (int): Keep every ``thin_by`` sample. Defaults to ``1``.
            show_progress (bool): Whether to display a progress bar. Defaults
                to ``False``.
            **kwargs: Additional keyword arguments passed to the proposal move.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostic
                information including acceptance rates.
        """
        # Use base class validation
        self._validate_mcmc_inputs(warmup, num_samples, thin_by, initial_state)

        if (initial_state.shape[0] != self.total_chains) and (initial_state.shape[1] != self.dim):
            raise ValueError("`inital_state` must have shape (total_chains, dim)")
        
        total_samples = warmup + num_samples * thin_by

        # Calculate batch size
        batch_size = calculate_batch_size(self.total_chains, self.dim, total_samples, batch_size)

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
            'accepts': jnp.zeros(self.total_chains),
        }
        
        # Initialize ensemble
        group1 = initial_state[:group1_size]
        group2 = initial_state[group1_size:]
        
        def body(carry, keys):
            group1, group2, diagnostics = carry

            # Construct Proposal
            group1_proposed, log_prob_group1 = self.move(group1, group2, keys[0], self.log_prob, **kwargs)
            group2_proposed, log_prob_group2 = self.move(group2, group1, keys[1], self.log_prob, **kwargs)

            # Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[0])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_prob_group2, keys[1])

            #### Logging diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            diagnostics['accepts'] += all_accepts
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])
            final_log_probs = jnp.concatenate([log_prob_group1, log_prob_group2])

            return (group1, group2, diagnostics), (final_states, final_log_probs, all_accepts)
                
        carry, self.backend = batched_scan(body, 
                             init_carry=(group1, group2, diagnostics), 
                             xs=keys, 
                             batch_size=batch_size, 
                             backend=self.backend,
                             show_progress=show_progress)
        _, _, diagnostics = carry

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics

        return self.get_chain(discard=warmup, thin=thin_by)