"""
High-level sampler implementations for hemcee.
"""

from typing import Callable, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp

from hemcee.moves.hamiltonian.hmc import hmc_move
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.adaptation.dual_averaging import init_da_parameters, init_da_state, da_cond_update, DAParameters, DAState
from hemcee.sampler_utils import accept_proposal, calculate_batch_size, batched_scan

class HamiltonianSampler:
    """Hamiltonian sampler with optional dual averaging.

    Attributes:
        total_chains (int): Total number of ensemble chains.
        dim (int): Dimensionality of the target distribution.
        log_prob (Callable): Vectorized log-probability function.
        grad_log_prob (Callable): Vectorized gradient of the log probability.
        step_size (float): Leapfrog step size.
        inv_mass_matrix (jnp.ndarray): Inverted mass matrix. Shape (dim, dim).
        L (int): Number of leapfrog steps per move.
        move (Callable): Proposal function updating each ensemble group.
        target_accept (float): Target acceptance probability for dual averaging.
        t0 (float): Dual averaging stabilizer parameter.
        mu (float): Dual averaging log step size offset.
        gamma (float): Dual averaging shrinkage parameter.
        kappa (float): Dual averaging adaptation rate.
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        step_size: float = 0.2,
        inv_mass_matrix: jnp.ndarray =  None,
        L: int = 10,
        move = hmc_move,
        storage_device: Optional[jax.Device] = None,
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
            storage_device = Optional[jax.Device]: Optional device to store intermediate results.
        """
        if (total_chains < 4):
            raise ValueError("`total_chains` must be at least 4 to form meaningful ensemble groups")
        
        self.total_chains = int(total_chains)
        self.dim = int(dim)

        self.move = move

        self.log_prob = jax.jit(jax.vmap(log_prob))
        self.grad_log_prob = jax.jit(jax.vmap(jax.grad(log_prob)))
        
        self.step_size = step_size
        self.L = L
        if inv_mass_matrix is None:
            self.inv_mass_matrix = jnp.eye(dim)
        else:    
            self.inv_mass_matrix = inv_mass_matrix


        # Dual Averaging parameters
        self.da_state = init_da_state(step_size)
        self.da_parameters = init_da_parameters(step_size)

        self.storage_device = storage_device

    def run_mcmc(self,
                 key: jax.random.PRNGKey,
                 initial_state: jnp.ndarray,
                 num_samples: int,
                 warmup: int = 1000,
                 thin_by=1,
                 batch_size: Optional[int] = None,
                 adapt_step_size: bool = True,
                 adapt_integration: bool = False,
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
            adapt_step_size (bool): Whether to adapt the step size via dual
                averaging. Defaults to ``True``.
            adapt_integration (bool): Whether to adapt integration settings
                using affine-invariant NUTS. Defaults to ``False``.
            show_progress (bool): Whether to display a progress bar. Defaults
                to ``False``.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and dual averaging state.
        """
        ########################################################
        # Check inputs for sanity
        ########################################################
        if adapt_integration:
            raise NotImplementedError("`adapt_integration=True` is not supported yet")
        
        if thin_by < 1:
            raise ValueError("`thinning` must 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")

        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")
        
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
            group1 = self._mcmc_warmup(key, group1, warmup, self.da_state, self.da_parameters, warmup_batch_size, thin_by, show_progress)

        # Main sampling
        # Statically sets step size & integration length from warmup
        step_size = jnp.exp(self.da_state.log_epsilon_bar)
        samples = self._mcmc_main(key, group1, num_samples, thin_by, step_size, self.inv_mass_matrix, self.L, main_batch_size, show_progress)

        return samples
    
    def _mcmc_warmup(self,
               key: jax.random.PRNGKey,
               group1: jnp.ndarray,
               warmup: int,
               da_state: DAState,
               da_parameters: DAParameters,
               batch_size: int,
               thin_by: int,
               show_progress: bool,
        ):
        """Run the warmup phase of the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Initial ensemble state for group 1 with shape
                ``(n_chains_per_group, dim)``.
            group2 (jnp.ndarray): Initial ensemble state for group 2 with shape
                ``(n_chains_per_group, dim)``.
            da_state (DAState): Initial dual averaging state.
                ``(total_chains, dim)``.
            da_parameters (DAParameters): Dual averaging parameters.
            warmup (int): Number of warmup iterations.
            show_progress (bool): Whether to display a progress bar.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and dual averaging state.
        """
        def body(carry, keys):
            group1, da_state, diagnostics = carry

            step_size = da_state.step_size

            #### Group 1 / Group 2 Update
            # Update group 1 using group 2 as complement
            group1_proposed, log_prob_group1 = self.move(group1, 
                step_size, self.inv_mass_matrix, self.L, 
                keys[0],
                self.log_prob, self.grad_log_prob, 
            )

            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[2])

            #### Combine diagnostics
            all_accepts = accept1
            current_accept_rate = jnp.mean(all_accepts)
            
            #### Dual Averaging
            da_state = da_cond_update(
                accept_prob=current_accept_rate,
                parameters=da_parameters,
                state=da_state,
            )
            
            #### Construct return state
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            return (group1, da_state, new_diagnostics), group1
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        
        n_keys = 4
        keys = jax.random.split(key, n_keys * warmup).reshape(warmup, n_keys, 2)
        
        carry, _ = batched_scan(body, 
                                init_carry=(group1, da_state, diagnostics), 
                                xs=keys, 
                                batch_size=batch_size, 
                                thin_by=thin_by,
                                storage_device=self.storage_device, show_progress=show_progress)
        group1, da_state, diagnostics = carry

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / warmup

        self.diagnostics_warmup = diagnostics
        self.da_state = da_state

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
        """Run the main sampling phase of the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Ensemble state for group 1 with shape
                ``(n_chains_per_group, dim)``.
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

            #### Group 1 / Group 2 Update
            # Update group 1 using group 2 as complement
            group1_proposed, log_prob_group1 = self.move(group1, 
                step_size, inv_mass_matrix, L,
                keys[0], 
                self.log_prob, self.grad_log_prob, 
            )
            #### Accept proposal?
            group1, accepts = accept_proposal(group1, group1_proposed, log_prob_group1, keys[0])

            #### Diagnostics
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + accepts,
            }
            
            #### Construct return state
            return (group1, new_diagnostics), group1
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        n_keys_per_iter = 4
        total_samples = num_samples * thin_by
        keys = jax.random.split(key, n_keys_per_iter * total_samples).reshape(total_samples, n_keys_per_iter, 2)

        carry, samples = batched_scan(body, 
                                      init_carry=(group1, diagnostics), 
                                      xs=keys, 
                                      batch_size=batch_size,
                                      thin_by=thin_by, 
                                      storage_device=self.storage_device, show_progress=show_progress)
        group1, diagnostics = carry

        # Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics

        return samples

class HamiltonianEnsembleSampler:
    """Hamiltonian ensemble sampler with optional dual averaging.

    Attributes:
        total_chains (int): Total number of ensemble chains.
        dim (int): Dimensionality of the target distribution.
        log_prob (Callable): Vectorized log-probability function.
        grad_log_prob (Callable): Vectorized gradient of the log probability.
        step_size (float): Leapfrog step size.
        L (int): Number of leapfrog steps per move.
        move (Callable): Proposal function updating each ensemble group.
        target_accept (float): Target acceptance probability for dual averaging.
        t0 (float): Dual averaging stabilizer parameter.
        mu (float): Dual averaging log step size offset.
        gamma (float): Dual averaging shrinkage parameter.
        kappa (float): Dual averaging adaptation rate.
    """

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        step_size: float = 0.2,
        L: int = 10,
        move = hmc_walk_move,
        storage_device: Optional[jax.Device] = None,
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
            storage_device = Optional[jax.Device]: Optional device to store intermediate results.
        """
        if (total_chains < 4):
            raise ValueError("`total_chains` must be at least 4 to form meaningful ensemble groups")
        
        self.total_chains = int(total_chains)
        self.dim = int(dim)

        self.move = move

        self.log_prob = jax.jit(jax.vmap(log_prob))
        self.grad_log_prob = jax.jit(jax.vmap(jax.grad(log_prob)))
        
        self.step_size = step_size
        self.L = L


        # Dual Averaging parameters
        self.da_state = init_da_state(step_size)
        self.da_parameters = init_da_parameters(step_size)

        self.storage_device = storage_device

    def run_mcmc(self,
                 key: jax.random.PRNGKey,
                 initial_state: jnp.ndarray,
                 num_samples: int,
                 warmup: int = 1000,
                 thin_by=1,
                 batch_size: Optional[int] = None,
                 adapt_step_size: bool = True,
                 adapt_integration: bool = False,
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
            adapt_step_size (bool): Whether to adapt the step size via dual
                averaging. Defaults to ``True``.
            adapt_integration (bool): Whether to adapt integration settings
                using affine-invariant NUTS. Defaults to ``False``.
            show_progress (bool): Whether to display a progress bar. Defaults
                to ``False``.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and dual averaging state.
        """
        ########################################################
        # Check inputs for sanity
        ########################################################        
        if adapt_integration:
            raise NotImplementedError("`adapt_integration=True` is not supported yet")
        
        if thin_by < 1:
            raise ValueError("`thinning` must 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")

        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")
        
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
            group1, group2 = self._mcmc_warmup(key, group1, group2, warmup, self.da_state, self.da_parameters, warmup_batch_size, thin_by, show_progress)
            print('Warmup complete.')

        # Main sampling
        # Statically sets step size & integration length from warmup
        step_size = jnp.exp(self.da_state.log_epsilon_bar)
        print('Starting main sampling...')
        samples = self._mcmc_main(key, group1, group2, num_samples, thin_by, step_size, self.L, main_batch_size, show_progress)
        print('Main sampling complete.')

        return samples
    
    def _mcmc_warmup(self,
               key: jax.random.PRNGKey,
               group1: jnp.ndarray,
               group2: jnp.ndarray,
               warmup: int,
               da_state: DAState,
               da_parameters: DAParameters,
               batch_size: int,
               thin_by: int,
               show_progress: bool,
        ):
        """Run the warmup phase of the Hamiltonian ensemble sampler.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            group1 (jnp.ndarray): Initial ensemble state for group 1 with shape
                ``(n_chains_per_group, dim)``.
            group2 (jnp.ndarray): Initial ensemble state for group 2 with shape
                ``(n_chains_per_group, dim)``.
            da_state (DAState): Initial dual averaging state.
                ``(total_chains, dim)``.
            warmup (int): Number of warmup iterations.

        Returns:
            tuple[jnp.ndarray, dict]: Post-warmup samples and diagnostics
            containing acceptance rates and dual averaging state.
        """
        def body(carry, keys):
            group1, group2, da_state, diagnostics = carry

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
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[2])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_prob_group2, keys[3])

            #### Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            current_accept_rate = jnp.mean(all_accepts)
            
            #### Dual Averaging
            da_state = da_cond_update(
                accept_prob=current_accept_rate,
                parameters=da_parameters,
                state=da_state,
            )
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            return (group1, group2, da_state, new_diagnostics), final_states
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        
        n_keys = 4
        keys = jax.random.split(key, n_keys * warmup).reshape(warmup, n_keys, 2)
        
        carry, samples = batched_scan(body, 
                                      init_carry=(group1, group2, da_state, diagnostics), 
                                      xs=keys, 
                                      batch_size=batch_size, 
                                      thin_by=thin_by,
                                      storage_device=self.storage_device, show_progress=show_progress)
        group1, group2, da_state, diagnostics = carry

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / warmup

        self.diagnostics_warmup = diagnostics
        self.da_state = da_state

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
            group1_proposed, log_prob_group1 = self.move(group1, group2, 
                step_size, keys[0],
                self.log_prob, self.grad_log_prob, 
                L
            )
            # Update group 2 using group 1 as complement  
            group2_proposed, log_prob_group2 = self.move(group2, group1, 
                step_size, keys[1],
                self.log_prob, self.grad_log_prob, 
                L
            )
            #### Accept proposal?
            group1, accept1 = accept_proposal(group1, group1_proposed, log_prob_group1, keys[0])
            group2, accept2 = accept_proposal(group2, group2_proposed, log_prob_group2, keys[1])

            #### Diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            new_diagnostics = {
                'accepts': diagnostics['accepts'] + all_accepts,
            }
            
            #### Construct return state
            final_states = jnp.concatenate([group1, group2])
            
            return (group1, group2, new_diagnostics), final_states
        
        diagnostics = {
            'accepts': jnp.zeros(self.total_chains),
        }
        n_keys_per_iter = 4
        total_samples = num_samples * thin_by
        keys = jax.random.split(key, n_keys_per_iter * total_samples).reshape(total_samples, n_keys_per_iter, 2)

        carry, samples = batched_scan(body, 
                                      init_carry=(group1, group2, diagnostics), 
                                      xs=keys, 
                                      batch_size=batch_size, 
                                      thin_by=thin_by,
                                      storage_device=self.storage_device, show_progress=show_progress)
        group1, group2, diagnostics = carry

        # Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples
        self.diagnostics_main = diagnostics

        return samples

class EnsembleSampler:
    """Affine-invariant ensemble sampler wrapper."""

    def __init__(
        self,
        total_chains: int,
        dim: int,
        log_prob: Callable,
        move = walk_move,
        storage_device: Optional[jax.Device] = None,
    ) -> None:
        """Initialise the vanilla ensemble sampler.

        Args:
            total_chains (int): Total number of chains in the ensemble.
            dim (int): Dimensionality of the target distribution.
            log_prob (Callable): Callable returning the log density of the
                target distribution.
            move (Callable): Proposal function used for ensemble updates.
            storage_device = Optional[jax.Device]: Optional device to store intermediate results.
        """
        self.total_chains = int(total_chains)
        if self.total_chains < 4:
            raise ValueError("`self.total_chains` must be at least 4 to form meaningful ensemble groups")

        self.dim = int(dim)

        self.log_prob = jax.jit(jax.vmap(log_prob))
        self.move = move

        self.storage_device = storage_device

    def run_mcmc(self,
               key: jax.random.PRNGKey,
               initial_state: jnp.ndarray,
               num_samples: int,
               warmup: int = 1000,
               thin_by=1,
               batch_size: Optional[int] = None,
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
        if thin_by < 1:
            raise ValueError("`thinning` must 1 or greater.")

        if warmup < 0:
            raise ValueError("`warmup` must be 0 or greater.")
            
        if num_samples < 0:
            raise ValueError("`num_samples` must be 0 or greater.")

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

            # Construt Proposal
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


            return (group1, group2, diagnostics), final_states
                
        carry, samples = batched_scan(body, 
                                      init_carry=(group1, group2, diagnostics), 
                                      xs=keys, 
                                      batch_size=batch_size, 
                                      thin_by=thin_by,
                                      storage_device=self.storage_device, show_progress=show_progress)
        _, _, diagnostics = carry

        # Return post-warmup samples (already thinned by batched_scan)
        warmup_thinned = warmup // thin_by if thin_by > 1 else warmup
        post_warmup_samples = samples[warmup_thinned:] if warmup > 0 else samples

        #### Logging
        diagnostics['acceptance_rate'] = diagnostics['accepts'] / total_samples

        self.diagnostics_main = diagnostics

        return post_warmup_samples
            
