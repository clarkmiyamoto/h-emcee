from typing import NamedTuple
import jax.numpy as jnp
import jax.lax as lax

class DAState(NamedTuple):
    step_size: jnp.ndarray
    H_bar: jnp.ndarray
    log_epsilon_bar: jnp.ndarray

def da_cond_update(iteration, warmup_length, accept_prob, *, target_accept, t0, mu, gamma, kappa, state: DAState):
    '''
    Update the dual averaging state
    '''
    def in_warmup(s: DAState):
        it = iteration
        H_bar_new = ((1.0 - 1.0/(it + 1 + t0)) * s.H_bar
                     + (target_accept - accept_prob) / (it + 1 + t0))
        log_eps = mu - (jnp.sqrt(it + 1.0)/gamma) * H_bar_new
        eta = (it + 1.0) ** (-kappa)
        log_eps_bar_new = eta * log_eps + (1.0 - eta) * s.log_epsilon_bar
        step_size_new = jnp.exp(log_eps)
        return DAState(step_size_new, H_bar_new, log_eps_bar_new)

    def after_warmup(s: DAState):
        return DAState(jnp.exp(s.log_epsilon_bar), s.H_bar, s.log_epsilon_bar)

    return lax.cond(iteration < warmup_length, in_warmup, after_warmup, state)

