from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from .sampler import BaseSampler
from .sampler_utils import calculate_batch_size, accept_proposal, batched_scan
from hemcee.moves.hamiltonian.hmc import hmc_move
from hemcee.adaptation.adapter import select_adapter, Adapter
from hemcee.adaptation.dual_averaging import DAParameters
from hemcee.adaptation.chees import ChEESParameters
from hemcee.backend.backend import Backend

class HamiltonianSampler(BaseSampler):
    """Hamiltonian sampler with optional dual averaging and ChEES adaptation.

    Attributes:
        total_chains (int): Total number of ensemble chains.
        dim (int): Dimensionality of the target distribution.
        log_prob (Callable): Vectorized log-probability function.
        grad_log_prob (Callable): Vectorized gradient of the log probability.
        step_size (float): Leapfrog step size.
        inv_mass_matrix (jnp.ndarray): Inverted mass matrix. Shape (dim, dim).
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
        step_size: float = 0.2,
        inv_mass_matrix: jnp.ndarray = None,
        L: int = 10,
        move = hmc_move,
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
            inv_mass_matrix (jnp.ndarray, optional): Inverted mass matrix. Shape (dim, dim).
            L (int): Number of leapfrog steps per proposal. Defaults to ``10``.
            move (Callable): Proposal function used for ensemble updates.
            backend: Backend for storing chain data.
            adapt_step_size (bool | DAParameters): Adapt step size of leapfrog integrator using Dual Averaging scheme.
                Defaults to yes, will adapt.
            adapt_length (bool | ChEESParameters): Adapt integration length of leapfrog integrator using ChEES scheme.
                Defaults to yes, will adapt.
        """
        # Initialize base class
        super().__init__(total_chains, dim, log_prob, move, backend)
        
        # Hamiltonian-specific parameters
        self.step_size = step_size
        self.L = L
        self.inv_mass_matrix = jnp.eye(dim) if inv_mass_matrix is None else inv_mass_matrix

        # Dual Averaging / ChEES Adapter
        self.adapter = select_adapter(adapt_step_size, adapt_length, self.step_size, self.L)
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
        """Run the Hamiltonian sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            initial_state (jnp.ndarray): Initial ensemble state with shape
                ``(total_chains, dim)``.
            num_samples (int): Number of post-warmup samples to retain.
            warmup (int): Number of warmup iterations. Defaults to ``1000``.
            thin_by (int): Keep every ``thin_by``
                sample. Defaults to ``1`` (no thinning).
            adapt_step_size (bool | DAParameters): Whether to adapt the step size via dual
                averaging. Defaults to ``True``.
            adapt_length (bool | ChEESParameters): Whether to adapt integration settings
                using ChEES. Defaults to ``True``.
            show_progress (bool): Whether to display a progress bar. Defaults
                to ``False``.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and adapter state.
        """
        ########################################################
        # Check inputs for sanity
        ########################################################                
        self._validate_mcmc_inputs(warmup, num_samples, thin_by, initial_state)
        
        ########################################################
        # MCMC Code
        ########################################################
        print(f"Using {self.total_chains} total chains")
        
        # Calculate batch sizes
        warmup_batch_size = calculate_batch_size(self.total_chains, self.dim, warmup, batch_size)
        main_batch_size = calculate_batch_size(self.total_chains, self.dim, num_samples * thin_by, batch_size)
        
        # Warmup
        # Contains dual averaging + ChEES updating
        group1 = initial_state
        if warmup > 0:
            print('Starting warmup...')
            group1 = self._mcmc_warmup(key, group1, warmup, 
                                       self.adapter, warmup_batch_size, thin_by, 
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
        self._mcmc_main(key, group1, num_samples, thin_by, step_size, self.inv_mass_matrix, L, main_batch_size, show_progress)
        print('Main sampling complete.')

        return self.get_chain(discard=warmup, thin=thin_by)

    def _mcmc_warmup(self,
               key: jax.random.PRNGKey,
               group1: jnp.ndarray,
               warmup: int,
               adapter: Adapter,
               batch_size: int,
               thin_by: int,
               show_progress: bool,
        ):
        """Run the warmup phase of the Hamiltonian sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Initial ensemble state with shape
                ``(total_chains, dim)``.
            adapter (Adapter): Adapter for step size and integration time adaptation.
            warmup (int): Number of warmup iterations.

        Returns:
            jnp.ndarray: Post-warmup ensemble state.
        """
        def body(carry, keys):
            group1, step_size, L, adapter_state, diagnostics = carry

            #### Group 1 Update
            # Update group 1 using HMC move
            group1_proposed, log_prob_group1 = self.move(group1, 
                step_size, self.inv_mass_matrix, L, 
                keys[0],
                self.log_prob, self.grad_log_prob, 
            )

            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[1])

            #### Combine diagnostics
            all_accepts = accept1
            current_accept_rate = jnp.mean(all_accepts)
            
            #### Adaptation
            adapter_state_new = adapter.update(adapter_state, current_accept_rate, group1)
            
            # Extract current adapted values
            step_size_new, L_new = adapter.value(adapter_state_new)
            
            #### Construct return state
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            # Compute log probabilities for current state
            log_prob_group1 = self.log_prob(group1)
            
            return (group1, step_size_new, L_new, adapter_state_new, new_diagnostics), (group1, log_prob_group1, accept1)
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        
        n_keys = 2
        keys = jax.random.split(key, n_keys * warmup).reshape(warmup, n_keys, 2)
        
        carry = batched_scan(body, 
                             init_carry=(group1, self.step_size, self.L, self.adapter_state, diagnostics), 
                             xs=keys, 
                             batch_size=batch_size, 
                             backend=self.backend,
                             show_progress=show_progress)
        group1, step_size_final, L_final, adapter_state_final, diagnostics = carry

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / warmup

        self.diagnostics_warmup = diagnostics
        self.adapter_state = adapter_state_final

        return group1

    def _mcmc_main(self,
                   key: jax.random.PRNGKey,
                   group1: jnp.ndarray,
                   num_samples: int,
                   thin_by: int,
                   step_size: float,
                   inv_mass_matrix: jnp.ndarray,
                   L: int,
                   batch_size: int,
                   show_progress: bool,
        ):
        """Run the main sampling phase of the Hamiltonian sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Ensemble state with shape
                ``(total_chains, dim)``.
            num_samples (int): Number of post-warmup samples to retain.
            thin_by (int): Keep every ``thin_by`` sample.
            step_size (float): Step size for the leapfrog integrator.
            inv_mass_matrix (jnp.ndarray): Inverted mass matrix. Shape (dim, dim).
            L (int): Number of leapfrog steps per proposal.

        Returns:
            samples (jnp.ndarray): Post-warmup samples with shape
                ``(num_samples, total_chains, dim)``.
        """

        ########################################################
        # Sampling Loop Body
        ########################################################
        def body(carry, keys):
            group1, diagnostics = carry

            #### Group 1 Update
            # Update group 1 using HMC move
            group1_proposed, log_prob_group1 = self.move(group1, 
                step_size, inv_mass_matrix, L,
                keys[0], 
                self.log_prob, self.grad_log_prob, 
            )
            #### Accept proposal?
            group1, accepts = accept_proposal(group1, group1_proposed, log_prob_group1, keys[1])

            #### Diagnostics
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + accepts,
            }
            
            #### Construct return state
            # Compute log probabilities for current state
            log_prob_group1 = self.log_prob(group1)
            
            return (group1, new_diagnostics), (group1, log_prob_group1, accepts)
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        n_keys_per_iter = 2
        total_samples = num_samples * thin_by
        keys = jax.random.split(key, n_keys_per_iter * total_samples).reshape(total_samples, n_keys_per_iter, 2)

        carry = batched_scan(body, 
                             init_carry=(group1, diagnostics), 
                             xs=keys, 
                             batch_size=batch_size,
                             backend=self.backend, 
                             show_progress=show_progress)
        group1, diagnostics = carry

        # Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics

