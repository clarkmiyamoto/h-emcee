"""End-to-end tests for NUTS sampler affine invariance."""
import pytest

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import hemcee
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move

from hemcee.tests.affine_invariance.tools import make_distributions_and_affine_transformation

dim = 2
total_chains = dim * 2

def test_affineInvariance_NUTS_EndToEnd():
    """
    Test NUTS sampler for affine invariance end-to-end.
    
    For an affine transformation phi(x) = A @ x + b:
    - Run sampler on original distribution starting from X
    - Run sampler on transformed distribution starting from phi(X)
    - Results should satisfy: phi(samples_original) = samples_transformed
    
    This uses the same random key for both runs to ensure bit-wise determinism.
    """
    for seed in [0, 1]:
        print(f"Running end-to-end NUTS test with seed {seed}")
        run_endtoend(seed)

def run_endtoend(seed):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    
    # Create distributions and affine transformation
    log_prob, log_prob_pushforward, A, b = make_distributions_and_affine_transformation(keys[0], dim)
    
    # Initialize starting positions
    initial_state = jax.random.normal(keys[1], shape=(total_chains, dim), dtype=jnp.float64)
    initial_state_transformed = jnp.einsum('ji,bi->bj', A, initial_state) + b[None, :]
    
    # Sampler parameters
    step_size = 0.1
    max_tree_depth = 5
    num_samples = 10
    warmup = 10
    
    ### Path A: Sample from original distribution, then transform
    sampler_original = hemcee.EnsembleNUTS(
        total_chains=total_chains,
        dim=dim,
        log_prob=log_prob,
        move=hmc_walk_move,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
    )
    
    samples_A, _ = sampler_original.run_mcmc(
        key=keys[2],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=1,
        batch_size=2,
        show_progress=False,
    )
    
    # Transform the samples: (num_samples, total_chains, dim)
    samples_A_transformed = jnp.einsum('ji,sbi->sbj', A, samples_A) + b[None, None, :]
    
    ### Path B: Sample from transformed distribution
    sampler_transformed = hemcee.EnsembleNUTS(
        total_chains=total_chains,
        dim=dim,
        log_prob=log_prob_pushforward,
        move=hmc_walk_move,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
    )
    
    samples_B, _ = sampler_transformed.run_mcmc(
        key=keys[2],  # Same key as Path A
        initial_state=initial_state_transformed,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=1,
        batch_size=2,
        show_progress=False,
    )
    
    ### Check affine invariance: transformed samples from A should equal samples from B
    print(f"  Max difference: {jnp.max(jnp.abs(samples_A_transformed - samples_B))}")
    assert jnp.allclose(samples_A_transformed, samples_B)

