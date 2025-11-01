import pytest
import jax
import jax.numpy as jnp

import hemcee
from hemcee.moves.hamiltonian.hmc import hmc_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move

from hemcee.adaptation.dual_averaging import DAParameters, DualAveragingAdapter
from hemcee.adaptation.chees import ChEESParameters, ChEESAdapter
from hemcee.adaptation.adapter import Adapter, NoOpAdapter, CompositeAdapter


seed = 0
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, num=30)  # More keys for parametrized tests

n_walkers = 4
n_dim = 1

def log_prob(x):
    return -0.5 * jnp.sum(x**2)

inital_state = jax.random.normal(keys[0], shape=(n_walkers, n_dim))

# Define moves and adapters for parametrization
# Ensemble moves use HamiltonianEnsembleSampler
MOVES = [
    ("walk", hmc_walk_move, "ensemble"),
    ("side", hmc_side_move, "ensemble"),
    ("hmc", hmc_move, "single"),
]

ADAPTER_CONFIGS = [
    ("both_enabled", True, True, CompositeAdapter),
    ("stepsize_only", True, False, DualAveragingAdapter),
    ("length_only", False, True, ChEESAdapter),
    ("no_adaptation", False, False, NoOpAdapter),
]


def test_default_adaptation_both_enabled():
    """
    Test default behavior of sampler, ensure dual averaging & ChEES is auto-enabled.
    """
    sampler_Default = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob)
    samples_fromDefault = sampler_Default.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5)

    sampler_BothEnabled = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob, adapt_step_size=True , adapt_length=True)
    samples_fromBothEnabled = sampler_BothEnabled.run_mcmc(keys[1], inital_state, num_samples=1, warmup=5,)

    assert isinstance(sampler_Default.adapter, CompositeAdapter), "Default adapter should be CompositeAdapter"
    assert isinstance(sampler_BothEnabled.adapter, CompositeAdapter), "Both enabled adapter should be CompositeAdapter"
    assert jnp.allclose(samples_fromDefault, samples_fromBothEnabled), "Outputs were not the same"

def test_stepsize_adaptation_disabled():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob, adapt_step_size=False , adapt_length=True)
    sampler.run_mcmc(keys[3], inital_state, num_samples=1, warmup=5,)

    assert isinstance(sampler.adapter, ChEESAdapter), 'Adapter is not `ChEESAdapter`'

def test_chees_adaptation_disable():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob, adapt_step_size=True , adapt_length=False,)
    sampler.run_mcmc(keys[4], inital_state,  num_samples=1, warmup=5,)

    assert isinstance(sampler.adapter, DualAveragingAdapter), 'Adapter is not `DualAveragingAdapter`'

def test_no_adaptation():
    sampler = hemcee.HamiltonianEnsembleSampler(n_walkers, n_dim, log_prob, adapt_step_size=False, adapt_length=False)
    sampler.run_mcmc(keys[5], inital_state, num_samples=1, warmup=5,)

    assert isinstance(sampler.adapter, NoOpAdapter), 'Adapter is not `NoOpAdapter`'


@pytest.mark.parametrize("move_name,move_func,sampler_type", MOVES)
@pytest.mark.parametrize("adapter_name,adapt_step_size,adapt_length,expected_adapter_class", ADAPTER_CONFIGS)
def test_adapter_with_move(move_name, move_func, sampler_type, adapter_name, adapt_step_size, adapt_length, expected_adapter_class):
    """
    Test every adapter type with every move type.
    
    Args:
        move_name: Name of the move (for test identification)
        move_func: The move function to test
        sampler_type: Type of sampler to use ("ensemble" or "single")
        adapter_name: Name of the adapter config (for test identification)
        adapt_step_size: Whether to adapt step size
        adapt_length: Whether to adapt integration length
        expected_adapter_class: Expected adapter class type
    """
    # Use a unique key for each test combination
    key_idx = hash((move_name, adapter_name)) % len(keys)
    test_key = keys[key_idx]
    
    # Use appropriate sampler based on move type
    if sampler_type == "ensemble":
        sampler = hemcee.HamiltonianEnsembleSampler(
            n_walkers, 
            n_dim, 
            log_prob, 
            move=move_func,
            adapt_step_size=adapt_step_size,
            adapt_length=adapt_length
        )
    else:  # sampler_type == "single"
        sampler = hemcee.HamiltonianSampler(
            n_walkers, 
            n_dim, 
            log_prob, 
            move=move_func,
            adapt_step_size=adapt_step_size,
            adapt_length=adapt_length
        )
    
    # Run sampler
    sampler.run_mcmc(test_key, inital_state, num_samples=1, warmup=5)
    
    # Verify adapter type
    assert isinstance(sampler.adapter, expected_adapter_class), \
        f'Adapter is not `{expected_adapter_class.__name__}` for move={move_name}, adapter={adapter_name}'




