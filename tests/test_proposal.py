import jax
import jax.numpy as jnp

from hemcee.proposal import accept_proposal


def test_accept_proposal_matches_expected_mask():
    key = jax.random.PRNGKey(0)
    current = jnp.zeros((2, 2))
    proposed = jnp.ones((2, 2))
    log_accept_prob = jnp.array([0.0, jnp.log(1e-8)])

    updated, accepts = accept_proposal(current, proposed, log_accept_prob, key)

    log_u = jnp.log(jax.random.uniform(key, shape=log_accept_prob.shape, minval=1e-10, maxval=1.0))
    expected_accepts = (log_u < log_accept_prob).astype(int)
    expected_updated = jnp.where(expected_accepts[:, None], proposed, current)

    assert jnp.allclose(updated, expected_updated)
    assert jnp.array_equal(accepts, expected_accepts)
