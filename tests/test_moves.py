import jax
import jax.numpy as jnp
import pytest

from hemcee.dual_averaging import DAState
from hemcee.moves.vanilla.side import side_move
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move


def gaussian_log_prob(x):
    return -0.5 * jnp.sum(x**2, axis=-1)


def gaussian_potential(x):
    return 0.5 * jnp.sum(x**2, axis=-1)


def gaussian_potential_grad(x):
    return x


def _build_groups(n_chains_per_group: int = 4, dim: int = 3):
    group1 = jnp.linspace(0.0, 1.0, num=n_chains_per_group * dim).reshape(
        n_chains_per_group, dim
    )
    group2 = jnp.linspace(1.0, 2.0, num=n_chains_per_group * dim).reshape(
        n_chains_per_group, dim
    )
    return group1, group2


def run_vanilla_move_test(move_fn, *, uses_potential: bool = False, **kwargs):
    key = jax.random.PRNGKey(0)
    group1, group2 = _build_groups()

    reference_key = jax.random.PRNGKey(0)

    target_fn = gaussian_potential if uses_potential else gaussian_log_prob

    proposed, log_accept = move_fn(group1, group2, key, target_fn, **kwargs)
    proposed_ref, log_accept_ref = move_fn(group1, group2, reference_key, target_fn, **kwargs)

    assert proposed.shape == group1.shape
    assert log_accept.shape == (group1.shape[0],)

    assert jnp.allclose(proposed, proposed_ref)
    assert jnp.allclose(log_accept, log_accept_ref)

    assert jnp.all(jnp.isfinite(proposed))
    assert jnp.all(jnp.isfinite(log_accept))


def _build_da_state(step_size: float = 0.25) -> DAState:
    step_size_arr = jnp.asarray(step_size)
    return DAState(
        step_size=step_size_arr,
        H_bar=jnp.asarray(0.0),
        log_epsilon_bar=jnp.log(step_size_arr),
    )


def run_hamiltonian_move_test(move_fn, **kwargs):
    key = jax.random.PRNGKey(0)
    group1, group2 = _build_groups()
    da_state = _build_da_state()

    reference_key = jax.random.PRNGKey(0)

    proposed, log_accept = move_fn(
        group1,
        group2,
        da_state,
        key,
        gaussian_potential,
        gaussian_potential_grad,
        **kwargs,
    )

    proposed_ref, log_accept_ref = move_fn(
        group1,
        group2,
        da_state,
        reference_key,
        gaussian_potential,
        gaussian_potential_grad,
        **kwargs,
    )

    assert proposed.shape == group1.shape
    assert log_accept.shape == (group1.shape[0],)

    assert jnp.allclose(proposed, proposed_ref)
    assert jnp.allclose(log_accept, log_accept_ref)

    assert jnp.all(jnp.isfinite(proposed))
    assert jnp.all(jnp.isfinite(log_accept))

    assert jnp.all(log_accept <= 1e-6)


@pytest.mark.parametrize(
    "move_fn, uses_potential",
    [
        (stretch_move, False),
        (side_move, False),
        (walk_move, True),
    ],
)
def test_vanilla_moves(move_fn, uses_potential):
    run_vanilla_move_test(move_fn, uses_potential=uses_potential)


@pytest.mark.parametrize("move_fn", [hmc_side_move, hmc_walk_move])
def test_hamiltonian_moves(move_fn):
    run_hamiltonian_move_test(move_fn, L=2)
