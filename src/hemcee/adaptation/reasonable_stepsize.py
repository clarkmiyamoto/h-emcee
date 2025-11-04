import jax
import jax.numpy as jnp
from typing import Callable

def find_reasonable_step_size(
    key: jax.random.PRNGKey,
    group1: jnp.ndarray,
    group2: jnp.ndarray,
    log_prob: Callable,
    grad_log_prob: Callable,
    init_step_size: float,
    proposal: Callable,
) -> float:
    """
    Finds a reasonable step size by tuning `init_step_size`. This function is used
    to avoid working with a too large or too small step size in HMC. This is based on
    the code from NumPyro.

    Args:
        key (jax.random.PRNGKey): Random key to be used as the source of randomness.
        log_prob (Callable): A callable to compute log probability.
        grad_log_prob (Callable): A callable to compute gradient of log probability.
        init_step_size (float): Initial step size to be tuned.
        leapfrog (Callable): Leapfrog integrator.
    Returns:
        A reasonable value for step size.

    """
    target_log_accept_prob = jnp.log(0.8)

    log_prob_group1 = log_prob(group1)
    log_prob_group2 = log_prob(group2)
    
    grad_log_prob_group1 = grad_log_prob(group1)
    grad_log_prob_group2 = grad_log_prob(group2)

    grad_log_prob_group1_pseudo_function = lambda x: grad_log_prob_group1
    grad_log_prob_group2_pseudo_function = lambda x: grad_log_prob_group2

    finfo = jnp.finfo(jnp.result_type(init_step_size))

    def _body_fn(state):
        step_size, _, direction, key_state = state
        key_state, key_proposal_1, key_proposal_2 = jax.random.split(key_state, 3)

        # Change order of magnitude of step-size
        step_size = (2.0**direction) * step_size

        # Find acceptance probabilities from new step-size
        _, log_accept_prob_1_new, _, _ = proposal(
            group1, group2, step_size, key_proposal_1, 
            log_prob, grad_log_prob_group1_pseudo_function, 
            L=1, log_prob_group1=log_prob_group1
        )
        _, log_accept_prob_2_new, _, _ = proposal(
            group2, group1, step_size, key_proposal_2, 
            log_prob, grad_log_prob_group2_pseudo_function, 
            L=1, log_prob_group1=log_prob_group2
        )

        # Compute average acceptance probability using harmonic mean
        all_log_accept_probs = jnp.concatenate([log_accept_prob_1_new, log_accept_prob_2_new])
        average_accept_prob = all_log_accept_probs.shape[0] - jax.scipy.special.logsumexp(-all_log_accept_probs)

        # Change direction based on average acceptance probability
        direction_new = jnp.where(target_log_accept_prob < average_accept_prob, 1, -1)

        return step_size, direction, direction_new, key_state

    def _cond_fn(state):
        step_size, last_direction, direction, _ = state
        # condition to run only if step_size is not too small or we are not decreasing step_size
        not_small_step_size_cond = (step_size > finfo.tiny) | (direction >= 0)
        # condition to run only if step_size is not too large or we are not increasing step_size
        not_large_step_size_cond = (step_size < finfo.max) | (direction <= 0)
        not_extreme_cond = not_small_step_size_cond & not_large_step_size_cond
        return not_extreme_cond & (
            (last_direction == 0) | (direction == last_direction)
        )

    step_size, _, _, _ = jax.lax.while_loop(_cond_fn, _body_fn, (init_step_size, 0, 0, key))
    
    return step_size