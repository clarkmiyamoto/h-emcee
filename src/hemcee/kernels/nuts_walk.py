import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import warnings
import time

from tqdm import tqdm

class AffineInvariantEnsembleNUTSSampler:
    """
    Affine Invariant Ensemble No-U-Turn Sampler (AIE-NUTS) implementation.
    
    Uses two groups of chains, where groups interact through 
    ensemble-based preconditioning. Each group uses the other as a 
    complement ensemble for momentum preconditioning.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 dim: int,
                 grad_log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
                 step_size: float = 0.1,
                 max_treedepth: int = 10,
                 target_accept: float = 0.8,
                 gamma: float = 0.05,
                 t0: float = 10.0,
                 kappa: float = 0.75,
                 beta: float = 1.0):
        """
        Initialize Affine Invariant Ensemble NUTS sampler.
        
        Args:
            log_prob_fn: Log probability function (dim,) -> ()
            dim: Problem dimension
            grad_log_prob_fn: Gradient function (dim,) -> (dim,). If None, uses JAX Autodiff.
            step_size: Initial step size
            max_treedepth: Maximum tree depth
            target_accept: Target acceptance probability for dual averaging
            gamma, t0, kappa: Dual averaging parameters
            beta: Ensemble interaction strength
        """
        self.log_prob_fn = jax.jit(jax.vmap(log_prob_fn)) # (batch_size, dim,) -> (batch_size,)
        if grad_log_prob_fn is None: # Defaults to JAX Autodiff
            self.grad_log_prob_fn = jax.jit(jax.vmap(jax.grad(log_prob_fn))) # (batch_size, dim,) -> (batch_size, dim,)
        else:
            self.grad_log_prob_fn = jax.jit(jax.vmap(grad_log_prob_fn)) # (batch_size, dim,) -> (batch_size, dim,)
        self.dim = dim
        self.step_size = step_size
        self.max_treedepth = max_treedepth
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.beta = beta
        
        # Dual averaging state (single step size for all chains)
        self.mu = jnp.log(10 * step_size)
        self.log_epsilon_bar = 0.0
        self.H_bar = 0.0
    
    def update_step_size(self, accept_prob: float, iteration: int, warmup_length: int):
        """Update step size using dual averaging."""
        if iteration < warmup_length:
            self.H_bar = ((1.0 - 1.0/(iteration + 1 + self.t0)) * self.H_bar + 
                         (self.target_accept - accept_prob) / (iteration + 1 + self.t0))
            
            log_epsilon = self.mu - jnp.sqrt(iteration + 1) / self.gamma * self.H_bar
            eta = (iteration + 1)**(-self.kappa)
            self.log_epsilon_bar = eta * log_epsilon + (1 - eta) * self.log_epsilon_bar
            
            self.step_size = jnp.exp(log_epsilon)
        else:
            self.step_size = jnp.exp(self.log_epsilon_bar)

        
    
    def compute_covariance_inv(self, complement_ensemble: jnp.ndarray) -> jnp.ndarray:
        """Compute inverse empirical covariance of complement ensemble."""
        n_complement = complement_ensemble.shape[0]
        
        # Center the complement ensemble
        complement_mean = jnp.mean(complement_ensemble, axis=0)
        centered = (complement_ensemble - complement_mean) / jnp.sqrt(n_complement)
        
        # Empirical covariance: C^T @ C
        emp_cov = jnp.dot(centered.T, centered)
        
        # Add regularization and invert
        reg = 1e-6 * jnp.eye(self.dim)
        try:
            return jnp.linalg.inv(emp_cov + reg)
        except jnp.linalg.LinAlgError:
            return jnp.linalg.pinv(emp_cov + reg)
    
    def ensemble_leapfrog(self, theta: jnp.ndarray, r: jnp.ndarray, 
                         epsilon: float, complement_ensemble: jnp.ndarray, 
                         direction: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Ensemble leapfrog step.
        
        Args:
            theta: Positions (n_chains_group, dim)
            r: Momenta (n_chains_group, n_complement) - ensemble momentum!
            epsilon: Step size
            complement_ensemble: Complement group positions (n_complement, dim)
            direction: +1 for forward, -1 for backward
        """
        n_complement = complement_ensemble.shape[0]
        
        # Compute centered complement
        complement_mean = jnp.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / jnp.sqrt(n_complement)
        
        if direction == 1:
            # Forward: half momentum, full position, half momentum
            grad = self.grad_log_prob_fn(theta)
            # Momentum update: grad projected onto complement ensemble
            r_half = r + 0.5 * epsilon * self.beta * jnp.dot(grad, centered_complement.T)
            
            # Position update: momentum projected back to position space
            theta_new = theta + epsilon * self.beta * jnp.dot(r_half, centered_complement)
            
            grad_new = self.grad_log_prob_fn(theta_new)
            r_new = r_half + 0.5 * epsilon * self.beta * jnp.dot(grad_new, centered_complement.T)
        else:
            # Backward: reverse the leapfrog
            grad = self.grad_log_prob_fn(theta)
            r_half = r - 0.5 * epsilon * self.beta * jnp.dot(grad, centered_complement.T)
            
            theta_new = theta - epsilon * self.beta * jnp.dot(r_half, centered_complement)
            
            grad_new = self.grad_log_prob_fn(theta_new)
            r_new = r_half - 0.5 * epsilon * self.beta * jnp.dot(grad_new, centered_complement.T)
        
        return theta_new, r_new
    
    def compute_uturn_criterion(self, theta_plus: jnp.ndarray, theta_minus: jnp.ndarray,
                               r_plus: jnp.ndarray, r_minus: jnp.ndarray,
                               complement_ensemble: jnp.ndarray,
                               cov_inv: jnp.ndarray) -> jnp.ndarray:
        """
        Compute U-turn criterion using ensemble-weighted metric.
        
        Returns boolean array: True means continue, False means stop.
        """
        n_complement = complement_ensemble.shape[0]
        delta_theta = theta_plus - theta_minus
        
        # Convert ensemble momentum to position space
        complement_mean = jnp.mean(complement_ensemble, axis=0)
        centered_complement = (complement_ensemble - complement_mean) / jnp.sqrt(n_complement)
        
        p_plus = jnp.dot(r_plus, centered_complement)
        p_minus = jnp.dot(r_minus, centered_complement)
        
        # Weighted inner products: delta_theta^T * cov_inv * p
        weighted_delta = jnp.dot(delta_theta, cov_inv)
        dot_plus = jnp.sum(weighted_delta * p_plus, axis=1)
        dot_minus = jnp.sum(weighted_delta * p_minus, axis=1)
        
        return (dot_plus >= 0) & (dot_minus >= 0)
    
    def build_tree(self, theta: jnp.ndarray, r: jnp.ndarray, u: jnp.ndarray,
                   direction: int, depth: int, epsilon: float,
                   complement_ensemble: jnp.ndarray, cov_inv: jnp.ndarray, key: jax.random.PRNGKey):
        """
        Build NUTS tree for ensemble of chains.
        
        Returns: theta_minus, r_minus, theta_plus, r_plus, theta_prime, 
                n_prime, s_prime, alpha_prime
        """
        if depth == 0:
            # Base case: single leapfrog step
            theta_prime, r_prime = self.ensemble_leapfrog(
                theta, r, epsilon, complement_ensemble, direction)
            
            # Compute log probabilities and energies
            log_prob_prime = self.log_prob_fn(theta_prime)
            log_prob_orig = self.log_prob_fn(theta)
            
            kinetic_prime = 0.5 * jnp.sum(r_prime**2, axis=1)
            kinetic_orig = 0.5 * jnp.sum(r**2, axis=1)
            
            joint_prime = log_prob_prime - kinetic_prime
            joint_orig = log_prob_orig - kinetic_orig
            
            # Slice condition
            log_u = jnp.log(jnp.clip(u, 1e-300, 1.0))
            n_prime = (log_u <= joint_prime).astype(int)
            s_prime = (joint_prime > log_u - 1000).astype(int)
            
            # Acceptance probability
            alpha_prime = jnp.minimum(1.0, jnp.exp(joint_prime - joint_orig))
            
            return (theta_prime, r_prime, theta_prime, r_prime,
                   theta_prime, n_prime, s_prime, alpha_prime)
        
        else:
            # Recursive case
            # Build first subtree
            (theta_minus, r_minus, theta_plus, r_plus,
             theta_prime, n_prime, s_prime, alpha_prime) = self.build_tree(
                theta, r, u, direction, depth - 1, epsilon, complement_ensemble, cov_inv, key)
            
            if jnp.any(s_prime == 1):
                # Build second subtree
                if direction == -1:
                    (theta_minus, r_minus, _, _, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_minus, r_minus, u, direction, depth - 1, epsilon,
                        complement_ensemble, cov_inv, key)
                else:
                    (_, _, theta_plus, r_plus, theta_double_prime,
                     n_double_prime, s_double_prime, alpha_double_prime) = self.build_tree(
                        theta_plus, r_plus, u, direction, depth - 1, epsilon,
                        complement_ensemble, cov_inv, key)
                
                # Multinomial sampling
                total_n = n_prime + n_double_prime
                valid = total_n > 0
                
                if jnp.any(valid):
                    prob = jnp.zeros(len(theta))
                    prob = prob.at[valid].set(n_double_prime[valid] / total_n[valid])
                    accept_mask = (jax.random.uniform(key, shape=(len(theta),)) < prob) & valid
                    theta_prime = theta_prime.at[accept_mask].set(theta_double_prime[accept_mask])

                    key = jax.random.split(key, 2)[1] # Updates key for new rng
                
                # Update acceptance probability
                alpha_prime = jnp.where(total_n > 0,
                                     (n_prime * alpha_prime + n_double_prime * alpha_double_prime) / total_n,
                                     alpha_prime)
                
                # Update stopping criterion
                continue_mask = self.compute_uturn_criterion(
                    theta_plus, theta_minus, r_plus, r_minus, complement_ensemble, cov_inv)
                s_prime = s_double_prime * continue_mask.astype(int)
                n_prime = total_n
            
            return (theta_minus, r_minus, theta_plus, r_plus,
                   theta_prime, n_prime, s_prime, alpha_prime)
    
    def nuts_step(self, theta: jnp.ndarray, complement_ensemble: jnp.ndarray,
                  epsilon: float, keys: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Single NUTS step for a group of chains.

        '''
        Args:
            theta: Current positions (n_chains, dim)
            complement_ensemble: Complement ensemble positions (n_complement, dim)
            epsilon: Step size
            keys: JAX random keys for reproducibility. Must be 4 unique keys.
        '''
        
        Returns: new_theta, acceptance_probs, tree_depths
        """
        n_chains = theta.shape[0]
        n_complement = complement_ensemble.shape[0]
        
        # Precompute covariance inverse
        cov_inv = self.compute_covariance_inv(complement_ensemble)
        
        # Sample momentum in ensemble space (n_chains, n_complement)
        r = jax.random.normal(keys[0], shape=(n_chains, n_complement))
        
        # Compute slice variable
        log_prob_current = self.log_prob_fn(theta)
        kinetic_current = 0.5 * jnp.sum(r**2, axis=1)
        joint_current = log_prob_current - kinetic_current
        u = jax.random.uniform(keys[1], shape=(n_chains,)) * jnp.exp(joint_current) # HACK: not stable
        
        # Initialize tree
        theta_minus = theta.copy()
        theta_plus = theta.copy()
        r_minus = r.copy()
        r_plus = r.copy()
        theta_new = theta.copy()
        
        depth = 0
        n = jnp.ones(n_chains, dtype=int)
        s = jnp.ones(n_chains, dtype=int)
        alpha_sum = jnp.zeros(n_chains)
        n_alpha = jnp.zeros(n_chains)
        
        # Track final tree depth for each chain individually
        # Initialize to -1 to indicate not yet terminated
        final_tree_depths = jnp.full(n_chains, -1, dtype=int)
        
        # Build tree until U-turn or max depth
        key_tree_search = keys[2]
        while jnp.any(s == 1) and depth < self.max_treedepth:
            # RNG for this iteration
            key_tree_search_iter = jax.random.split(key_tree_search, 2) # Updates key for new rng
            key_tree_search = key_tree_search_iter[0] # Updates key for new rng
            # Store which chains are active before this iteration
            active_before = (s == 1).copy()
            
            # Choose direction
            direction = jax.random.choice(key_tree_search_iter[1], jnp.array([1, -1]))
            
            # Expand tree
            if direction == -1:
                (theta_minus, r_minus, _, _, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_minus, r_minus, u, direction, depth, epsilon,
                    complement_ensemble, cov_inv, key_tree_search_iter[2])
            else:
                (_, _, theta_plus, r_plus, theta_prime,
                 n_prime, s_prime, alpha_prime) = self.build_tree(
                    theta_plus, r_plus, u, direction, depth, epsilon,
                    complement_ensemble, cov_inv, key_tree_search_iter[2])
            
            # Update positions
            if jnp.any(s_prime == 1):
                prob = jnp.minimum(1.0, n_prime / n)
                accept_mask = (jax.random.uniform(key_tree_search_iter[3], shape=(n_chains,)) < prob) & (s_prime == 1)
                theta_new = theta_new.at[accept_mask].set(theta_prime[accept_mask])
            
            # Track acceptance probabilities
            valid_alpha = n_prime > 0
            if jnp.any(valid_alpha):
                alpha_sum = alpha_sum.at[valid_alpha].add(n_prime[valid_alpha] * alpha_prime[valid_alpha])
                n_alpha = n_alpha.at[valid_alpha].add(n_prime[valid_alpha])
            
            # Update counts and stopping
            n += n_prime
            
            # Increment depth BEFORE checking termination
            depth += 1
            
            # Record depth for chains that terminate in this iteration
            # (were active before, but stopped now)
            newly_stopped = active_before & (s_prime == 0)
            final_tree_depths = final_tree_depths.at[newly_stopped].set(depth)
            
            # Update stopping criterion
            s = s_prime
        
        # For chains still active at max depth, record max depth
        still_active = (s == 1)
        final_tree_depths = final_tree_depths.at[still_active].set(depth)
        
        # Handle chains that stopped in first iteration (should have depth 1, not -1)
        never_updated = (final_tree_depths == -1)
        final_tree_depths = final_tree_depths.at[never_updated].set(1)
        
        # Compute final acceptance probabilities
        accept_probs = jnp.where(n_alpha > 0, alpha_sum / n_alpha, 0.0)
        
        return theta_new, accept_probs, final_tree_depths
    
    def sample(self, theta_init: jnp.ndarray, num_samples: int,
               total_chains: int = None, warmup: int = 1000, 
               adapt_step_size: bool = True,
               key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[jnp.ndarray, dict]:
        """
        Run Ensemble NUTS sampling.
        
        Args:
            theta_init: Initial position (will be replicated)
            num_samples: Number of post-warmup samples
            total_chains: Total number of chains to use (default: 2*dim, minimum: 4)
            warmup: Number of warmup samples
            adapt_step_size: Whether to adapt step size
            key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
        Returns:
            samples: (total_samples, total_chains, dim) array
            diagnostics: Dictionary of diagnostic information
        """
        if total_chains is None:
            total_chains = max(4, 2 * self.dim)
            
        if total_chains < 4:
            raise ValueError("total_chains must be at least 4 to form meaningful ensemble groups")
            
        total_samples = warmup + num_samples
        
        # Split chains into two groups
        group1_size = total_chains // 2
        group2_size = total_chains - group1_size
        
        print(f"Using {total_chains} total chains: Group 1 ({group1_size}), Group 2 ({group2_size})")
        
        # Keys for RNG
        total_rng_calls = 2 * total_samples * 4
        keys = jax.random.split(key, total_rng_calls).reshape(2 * total_samples, 4, 2)

        # Initialize ensemble
        theta_ensemble = jnp.tile(theta_init, (total_chains, 1))
        theta_ensemble += 0.1 * jax.random.normal(key, shape=(total_chains, self.dim))
        
        # Split into groups
        group1 = theta_ensemble[:group1_size]
        group2 = theta_ensemble[group1_size:]
        
        # Storage
        samples = jnp.zeros((total_samples, total_chains, self.dim))
        accept_probs_history = []
        tree_depths_history = []
        step_sizes_history = []
        
        for i in tqdm(range(total_samples)):
            # Store current state
            samples = samples.at[i, :group1_size].set(group1)
            samples = samples.at[i, group1_size:].set(group2)
            
            # Update group 1 using group 2 as complement
            group1, accept1, depths1 = self.nuts_step(group1, group2, self.step_size, keys[i])
            
            # Update group 2 using group 1 as complement  
            group2, accept2, depths2 = self.nuts_step(group2, group1, self.step_size, keys[i+1])
            
            # Combine diagnostics
            all_accepts = jnp.concatenate([accept1, accept2])
            all_depths = jnp.concatenate([depths1, depths2])
            
            # Adapt step size
            if adapt_step_size:
                mean_accept = jnp.mean(all_accepts)
                self.update_step_size(mean_accept, i, warmup)
            
            # Store diagnostics
            accept_probs_history.append(all_accepts)
            tree_depths_history.append(all_depths)
            step_sizes_history.append(self.step_size)
            
            # Progress
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i+1}/{total_samples}, "
                      f"Mean accept: {jnp.mean(all_accepts):.3f}, "
                      f"Step size: {self.step_size:.4f}")
        
        # Return post-warmup samples
        post_warmup_samples = samples[warmup:] if warmup > 0 else samples
        
        diagnostics = {
            'accept_probs': jnp.array(accept_probs_history),
            'tree_depths': jnp.array(tree_depths_history),
            'step_sizes': jnp.array(step_sizes_history),
            'mean_accept_prob': jnp.mean(jnp.array(accept_probs_history[warmup:]) if warmup > 0 else jnp.array(accept_probs_history)),
            'final_step_size': self.step_size,
            'warmup_samples': warmup,
            'total_chains': total_chains,
            'group_sizes': (group1_size, group2_size)
        }
        
        return post_warmup_samples, diagnostics

def test_ensemble_nuts():
    import time
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    
    """Test Ensemble NUTS on high-dimensional Gaussian with different chain counts."""
    print("=== Testing Flexible Ensemble NUTS Sampler ===")
    
    # Problem setup
    dim = 20
    n_samples = 5000
    warmup = 1000
    
    np.random.seed(42)
    cond_number = 10000
    eigenvals = 0.1 * np.linspace(1, cond_number, dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    precision = Q @ np.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)
    precision = jnp.array(precision)
    
    true_mean = jnp.ones(dim)
    initial = jnp.ones(dim)
    
    # Vectorized target functions
    def log_prob_fn(x):
        """Vectorized negative log density (potential energy)"""        
        # Vectorized operation for all samples
        centered = x - true_mean
        result = - 0.5 * jnp.einsum('j,jk,k->', centered, precision, centered)
            
        return result
    
    # Test with different numbers of chains
    chain_counts = [2*dim]
    
    for total_chains in chain_counts:
        print(f"\n--- Testing with {total_chains} chains ---")
        
        # Run Ensemble NUTS
        start_time = time.time()
        sampler = AffineInvariantEnsembleNUTSSampler(
            log_prob_fn=log_prob_fn,
            max_treedepth=3,
            dim=dim,
            
            step_size=1.0,
            target_accept=0.8,
            beta=1.0
        )
        
        samples, diagnostics = sampler.sample(
            initial, num_samples=100, warmup=warmup, 
            total_chains=total_chains, adapt_step_size=True
        )

        elapsed_time = time.time() - start_time
        
        # Analysis
        flat_samples = samples.reshape(-1, dim)
        sample_mean = np.mean(flat_samples, axis=0)
        mean_error = np.linalg.norm(sample_mean - true_mean)
        
        print(f"Total chains: {diagnostics['total_chains']}")
        print(f"Group sizes: {diagnostics['group_sizes']}")
        print(f"Samples shape: {samples.shape}")
        print(f"Mean error: {mean_error:.4f}")
        print(f"Time: {elapsed_time:.1f}s")
        print(f"Mean acceptance: {diagnostics['mean_accept_prob']:.3f}")
        print(f"Final step size: {diagnostics['final_step_size']:.4f}")
        print(f"Average tree depth: {np.mean(diagnostics['tree_depths']):.2f}")
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = test_ensemble_nuts()