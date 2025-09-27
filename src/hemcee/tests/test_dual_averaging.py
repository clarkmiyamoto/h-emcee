import jax.numpy as jnp

from hemcee.adaptation.dual_averaging import DAState, DAParameters, da_cond_update

def test_da_cond_update_in_warmup_branch():
    da_state = DAState(
        iteration=0,
        step_size=jnp.array(0.5),
        H_bar=jnp.array(0.0),
        log_epsilon_bar=jnp.array(jnp.log(0.5)),
    )
    da_parameters = DAParameters(
        target_accept=0.8,
        t0=10.0,
        mu=0.05,
        gamma=0.05,
        kappa=0.75,
    )

    updated = da_cond_update(
        accept_prob=0.2,
        parameters=da_parameters,
        state=da_state,
    )

    it = 0
    H_bar_new = ((1.0 - 1.0 / (it + 1 + 10.0)) * da_state.H_bar
                 + (0.8 - 0.2) / (it + 1 + 10.0))
    log_eps = 0.05 - (jnp.sqrt(it + 1.0) / 0.05) * H_bar_new
    eta = (it + 1.0) ** (-0.75)
    log_eps_bar_new = eta * log_eps + (1.0 - eta) * da_state.log_epsilon_bar
    expected = DAState(
        iteration=it + 1, 
        step_size=jnp.exp(log_eps), 
        H_bar=H_bar_new, 
        log_epsilon_bar=log_eps_bar_new,
    )

    assert jnp.allclose(updated.step_size, expected.step_size)
    assert jnp.allclose(updated.H_bar, expected.H_bar)
    assert jnp.allclose(updated.log_epsilon_bar, expected.log_epsilon_bar)