import jax.numpy as jnp

from hemcee.dual_averaging import DAState, da_cond_update


def test_da_cond_update_in_warmup_branch():
    state = DAState(
        step_size=jnp.array(0.5),
        H_bar=jnp.array(0.0),
        log_epsilon_bar=jnp.array(jnp.log(0.5)),
    )

    updated = da_cond_update(
        iteration=0,
        warmup_length=5,
        accept_prob=0.2,
        target_accept=0.8,
        t0=10.0,
        mu=0.05,
        gamma=0.05,
        kappa=0.75,
        state=state,
    )

    it = 0
    H_bar_new = ((1.0 - 1.0 / (it + 1 + 10.0)) * state.H_bar
                 + (0.8 - 0.2) / (it + 1 + 10.0))
    log_eps = 0.05 - (jnp.sqrt(it + 1.0) / 0.05) * H_bar_new
    eta = (it + 1.0) ** (-0.75)
    log_eps_bar_new = eta * log_eps + (1.0 - eta) * state.log_epsilon_bar
    expected = DAState(jnp.exp(log_eps), H_bar_new, log_eps_bar_new)

    assert jnp.allclose(updated.step_size, expected.step_size)
    assert jnp.allclose(updated.H_bar, expected.H_bar)
    assert jnp.allclose(updated.log_epsilon_bar, expected.log_epsilon_bar)


def test_da_cond_update_after_warmup_branch():
    state = DAState(
        step_size=jnp.array(0.5),
        H_bar=jnp.array(0.1),
        log_epsilon_bar=jnp.array(jnp.log(0.3)),
    )

    updated = da_cond_update(
        iteration=5,
        warmup_length=5,
        accept_prob=0.2,
        target_accept=0.8,
        t0=10.0,
        mu=0.05,
        gamma=0.05,
        kappa=0.75,
        state=state,
    )

    expected = DAState(jnp.exp(state.log_epsilon_bar), state.H_bar, state.log_epsilon_bar)

    assert jnp.allclose(updated.step_size, expected.step_size)
    assert jnp.allclose(updated.H_bar, expected.H_bar)
    assert jnp.allclose(updated.log_epsilon_bar, expected.log_epsilon_bar)
