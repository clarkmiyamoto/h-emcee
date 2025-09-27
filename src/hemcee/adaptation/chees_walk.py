from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple, Optional

import jax
import jax.numpy as jnp

import numpy as np
import corner
import matplotlib.pyplot as plt

Array = jax.Array
LogProbFn = Callable[[Array], Tuple[Array, Array]]  # returns (logp, grad)

pyte
# ----------------------- Utilities -----------------------

def welford_ema_mean(mu: Array, x: Array, momentum: float) -> Array:
    """Exponential moving average of mean across batch dimension 0.
    Args:
        mu: (..., d) previous mean
        x: (B, ..., d) batch of samples to incorporate
        momentum: in [0,1). new_mu = momentum*mu + (1-momentum)*mean(x)
    """
    batch_mean = jnp.mean(x, axis=0)
    return momentum * mu + (1.0 - momentum) * batch_mean


def compute_L_from_positions(X: Array, lam: float = 1e-6) -> Array:
    """Return Cholesky factor L of an SPD metric built from positions X.
    Uses centered empirical covariance: Sigma = (X_c^T X_c)/(n-1) + lam*I.
    Args:
        X: (N, d) positions used to define the metric (e.g., complement group)
        lam: ridge added to stabilize / ensure SPD
    Returns:
        L: (d, d) lower-triangular with Sigma = L @ L.T
    """
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    n = X.shape[0]
    # If n==1, fall back to isotropic metric
    if n <= 1:
        d = X.shape[1]
        return jnp.sqrt(lam) * jnp.eye(d)
    Sigma = (Xc.T @ Xc) / (n - 1) + lam * jnp.eye(X.shape[1])
    return jnp.linalg.cholesky(Sigma)


def maha_sq(x_minus_mu: Array, L: Array) -> Array:
    """Mahalanobis squared norm ||x-mu||_{Sigma^{-1}}^2 using Sigma=L L^T.
    Works with batched inputs x_minus_mu of shape (B, d) by solving on the
    transposed RHS to satisfy triangular_solve's (..., m, n) requirement.
    """
    # x_minus_mu: (B, d) or (d,)
    if x_minus_mu.ndim == 1:
        y = jax.scipy.linalg.solve_triangular(L, x_minus_mu, lower=True)
        return jnp.dot(y, y)
    # Batched: solve on RHS with shape (d, B), then transpose back
    yT = jax.scipy.linalg.solve_triangular(L, x_minus_mu.T, lower=True)  # (d, B)
    y = yT.T  # (B, d)
    return jnp.sum(y * y, axis=-1)


def maha_dot(a: Array, b: Array, L: Array) -> Array:
    """Metric inner product <a, b>_{Sigma^{-1}} with Sigma=L L^T.
    Supports a,b of shape (B, d) by solving on transposed RHS.
    Returns (B,) for batched inputs or scalar for vectors.
    """
    if a.ndim == 1:
        y = jax.scipy.linalg.solve_triangular(L, a, lower=True)
        z = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
        return jnp.dot(z, b)
    # Batched: (B,d)
    yT = jax.scipy.linalg.solve_triangular(L, a.T, lower=True)            # (d,B)
    zT = jax.scipy.linalg.solve_triangular(L.T, yT, lower=False)          # (d,B)
    z = zT.T                                                              # (B,d)
    return jnp.sum(z * b, axis=-1)


# ----------------------- Leapfrog (Euclidean dynamics) -----------------------

def leapfrog_walk_move(q: jnp.ndarray, 
                       p: jnp.ndarray, 
                       grad_fn: Callable, 
                       beta_eps: float, 
                       L: jnp.ndarray,
                       centered: jnp.ndarray):
    '''
    Args:
        q: Shape (n_chains_per_group, dim)
        p: Shape (n_chinas_per_group, n_chains_per_group)
        grad_fn: Gradient of log probabiltiy vectorized. Maps (batch_size, dim) -> (batch_size, dim)
        beta_eps: beta times step size (step_size)
        L: Number of steps
        centered: Shape (n_chains_per_group, dim)
    '''
    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0) 

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T) # Shape (n_chains_per_group, n_chains_per_group)
   
    
    # Use a fixed maximum number of steps and mask extra steps
    L_MAX = 200
    Lsteps = jnp.minimum(L, L_MAX)
    
    def integrate_with_masking(q, p, L_vals): # HACK: This is a hack to get the correct shape for the L values
        """Integrate with masking for different L values per chain"""
        def step_fn(carry, step_idx):
            q, p, L_vals = carry
            
            # Only update if step_idx < L_vals for each chain
            mask = (step_idx < L_vals)[:, None]  # (n_chains, 1)
            
            q += beta_eps * jnp.dot(p, centered) * mask  # Shape (n_chains_per_group, dim)
            
            grad = grad_fn(q) # Shape (n_chains_per_group, dim)
            grad = jnp.nan_to_num(grad, nan=0.0)
            p -= beta_eps * jnp.dot(grad, centered.T) * mask
            
            return (q, p, L_vals), None
        
        (q, p, _), _ = jax.lax.scan(step_fn, (q, p, L_vals), jnp.arange(L_MAX), length=L_MAX)
        return q, p
    
    q, p = integrate_with_masking(q, p, Lsteps)

    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0)

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

    return q, p

# ----------------------- HMC step (unchanged) -----------------------

def hmc_walk_move(
    key: jax.random.PRNGKey,
    group1: jnp.ndarray, group2: jnp.ndarray,
    logprob_and_grad: LogProbFn,
    eps: float,
    T: float,):
    '''
    Hamiltonian Walk Move (HWM) sampler implementation using JAX.
    Algorithm (3) in https://arxiv.org/pdf/2505.02987.

    Args:
        group1: Proposal group. Shape (n_chains_per_group, dim)
        group2: Complement group. Shape (n_chains_per_group, dim)
        da_state: Dual Averaging State
        key: JAX random key
        n_chains_per_group: Number of chains per group (total chains = 2 * n_chains_per_group).
        potential_func_vmap: Potential function vectorized
        grad_potential_func_vmap: Gradient of potential function vectorized
        L: Number of leapfrog steps
    '''
    n_chains_per_group = int(group1.shape[0])

    # [NEW] Jitter integration time
    # Prevents chains from having similar integration times
    u = jax.random.uniform(key, shape=(n_chains_per_group, 1))  # per-chain jitter
    t = u * T
    Lsteps = jnp.maximum(1, jnp.round(t / eps).astype(int))  # (B,1)
    Lsteps = jnp.squeeze(Lsteps, axis=-1)  # (B,)

    # Cap max number of steps to avoid explosive trajectories
    L_MAX = 200
    Lsteps = jnp.minimum(Lsteps, L_MAX)

    def grad_logprob(x: Array) -> Array:
        _, g = logprob_and_grad(x)
        return g


    key_momentum, key_accept = jax.random.split(key)
    centered2 = (group2 - jnp.mean(group2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)
    centered2 = jnp.nan_to_num(centered2, nan=0.0, posinf=0.0, neginf=0.0)
    momentum = jax.random.normal(key_momentum, shape=(n_chains_per_group, n_chains_per_group))


    group1_proposed, momentum_proposed = leapfrog_walk_move(
        group1, momentum, grad_logprob, eps, Lsteps, centered2
    )
    current_U, _ = logprob_and_grad(group1) # Shape (n_chains_per_group,)
    current_K = 0.5 * jnp.sum(momentum**2, axis=1) # Shape (n_chains_per_group,)
    
    proposed_U, _ = logprob_and_grad(group1_proposed)
    proposed_K = 0.5 * jnp.sum(momentum_proposed**2, axis=1)

    dH = current_U + current_K - (proposed_U + proposed_K)

    log_accept_prob1 = jnp.minimum(0.0, -dH)
    accept_prob = jnp.where(dH > 0, jnp.exp(-dH), jnp.ones_like(dH))

    return group1_proposed, log_accept_prob1, momentum_proposed, momentum, accept_prob, jnp.squeeze(t, -1)

def accept_proposal(
    key: jax.random.PRNGKey,
    q: Array,
    q_proposed: Array,
    accept_prob: Array,
) -> Array:
    '''
    Accept or reject a proposal in metropolis-hastings step.
    Log space is used for numerical stability.
    '''
    u = jax.random.uniform(key, shape=accept_prob.shape)
    accept = (u < accept_prob).astype(q_proposed.dtype)
    q_new = accept[:, None] * q_proposed + (1 - accept)[:, None] * q
    return q_new


# ----------------------- ChEES gradient (metric-aware) -----------------------

def chees_grad_T(
    q0: Array, q1: Array, p1: Array, t: Array, mu: Array, accept: Array,
    L_metric: Optional[Array] = None,
) -> Array:
    """Per-iteration stochastic gradient estimator for d/dT of ChEES objective.
    If L_metric is None, uses Euclidean metric. Otherwise uses Mahalanobis
    norms/inner products with Sigma = L_metric @ L_metric.T.
    Args:
        q0: (B, d) start positions
        q1: (B, d) proposed end positions
        p1: (B, d) end momenta (after final half step)
        t:  (B,)  actual integration times used (u*T per chain)
        mu: (d,)   running mean used in criterion
        accept: (B,) accept indicators or probabilities
        L_metric: (d,d) lower-triangular Cholesky of metric covariance (optional)
    Returns:
        scalar gradient estimate wrt T (averaged over batch)
    """
    dq0 = q0 - mu  # (B, d)
    dq1 = q1 - mu  # (B, d)

    if L_metric is None:
        qnorm_diff = jnp.sum(dq1 * dq1, axis=-1) - jnp.sum(dq0 * dq0, axis=-1)  # (B,)
        inner = jnp.sum(dq1 * p1, axis=-1)  # (B,)
    else:
        qnorm_diff = maha_sq(dq1, L_metric) - maha_sq(dq0, L_metric)
        inner = maha_dot(dq1, p1, L_metric)

    # Gradient estimator (metric-aware): t * (Δ||·||^2) * <dq1, p1>_{Sigma^{-1}}
    g_i = t * qnorm_diff * inner  # (B,)
    g_i = g_i * accept  # damp by accept prob/indicator
    return jnp.mean(g_i)


# ----------------------- Adaptation loop -----------------------

@dataclass
class ChEESConfig:
    n_chains: int = 32
    n_warmup: int = 1_000
    n_samples: int = 1_000
    dim: int = 2
    eps: float = 0.1
    T_init: float = 1.0
    T_min: float = 0.25
    T_max: float = 10.0
    lr_T: float = 5e-3
    mu_momentum: float = 0.9  # EMA momentum for running mean

    # --- New: metric options for ChEES objective only ---
    chees_metric: str = "euclidean"  # "euclidean" or "mahalanobis"
    metric_lam: float = 1e-6          # ridge for covariance metric
    # Optional callback: given current state, return positions (N,d) to define metric
    positions_for_metric: Optional[Callable[["ChEESState"], Array]] = None


class ChEESState(NamedTuple):
    key: Array
    q1: Array           # (B,d)
    q2: Array           # (B,d)
    mu: Array          # (d,)
    T: Array           # scalar array for JIT friendliness


def run_chees(
    key: Array,
    logprob_and_grad: LogProbFn,
    cfg: ChEESConfig,
    q0: Array = None,
) -> Tuple[ChEESState, Array, Array]:
    """Run ChEES-adapted HMC and return samples.
    Only the ChEES criterion/gradient are metric-aware; dynamics remain Euclidean.
    Args:
        key: PRNG key
        logprob_and_grad: function(q)->(logp, grad), batched over q
        cfg: config
        q0: optional initial positions (B,d); if None, sample N(0,I)
    Returns:
        final_state, samples (n_samples, B, d), accept_rates (n_warmup+n_samples,)
    """
    B, d = cfg.n_chains, cfg.dim
    if q0 is None:
        key, kq = jax.random.split(key)
        q = jax.random.normal(kq, shape=(B, d))
    else:
        q = q0

    # Initialize mu as batch mean of q
    mu = jnp.mean(q, axis=0)
    T = jnp.asarray(cfg.T_init)

    state = ChEESState(key, q, q, mu, T)

    def metric_L_for_state(state: ChEESState) -> Optional[Array]:
        if cfg.chees_metric == "euclidean":
            return None
        # mahalanobis: pick positions to define metric
        if cfg.positions_for_metric is not None:
            X = cfg.positions_for_metric(state)  # (N,d)
        else:
            # Fallback: use both groups' current chains as proxy
            X = jnp.concatenate([state.q1, state.q2], axis=0)
        return compute_L_from_positions(X, lam=cfg.metric_lam)

    def one_step_warmup(state: ChEESState, _):
        key = state.key
        key, k1, k2, ka1, ka2 = jax.random.split(key, 5)

        # Propose for both groups
        q1_prop, loga1, p1_prop_group, p1_0_group, a_prob1, t1 = hmc_walk_move(k1, state.q1, state.q2, logprob_and_grad, cfg.eps, state.T)
        q2_prop, loga2, p2_prop_group, p2_0_group, a_prob2, t2 = hmc_walk_move(k2, state.q2, state.q1, logprob_and_grad, cfg.eps, state.T)

        n = state.q1.shape[0]
        centered2 = (state.q2 - jnp.mean(state.q2, axis=0)[None, :]) / jnp.sqrt(n)
        centered1 = (state.q1 - jnp.mean(state.q1, axis=0)[None, :]) / jnp.sqrt(n)

        # Map group-space momenta to q-space for ChEES gradient
        p1_q = p1_prop_group @ centered2  # (n, d)
        p2_q = p2_prop_group @ centered1  # (n, d)

        # Accept/reject
        q1_new = accept_proposal(ka1, state.q1, q1_prop, a_prob1)
        q2_new = accept_proposal(ka2, state.q2, q2_prop, a_prob2)

        q_new_concat = jnp.concatenate([q1_new, q2_new], axis=0)

        # Update mu
        mu_new = welford_ema_mean(state.mu, q_new_concat, cfg.mu_momentum)

        # --- Metric-aware ChEES gradient wrt T ---
        L_metric = metric_L_for_state(state)
        q0_concat = jnp.concatenate([state.q1, state.q2], axis=0)
        q1_concat = jnp.concatenate([q1_prop, q2_prop], axis=0)
        p_concat = jnp.concatenate([p1_q, p2_q], axis=0)
        t_concat = jnp.concatenate([t1, t2], axis=0)
        a_concat = jnp.concatenate([a_prob1, a_prob2], axis=0)
        gT = chees_grad_T(q0_concat, q1_concat, p_concat, t_concat, state.mu, a_concat, L_metric=L_metric)
        gT = jnp.nan_to_num(gT, nan=0.0, posinf=0.0, neginf=0.0)

        # Gradient ascent on log T (keep T positive, clamp bounds)
        logT = jnp.log(state.T)
        logT = logT + cfg.lr_T * gT
        T_new = jnp.clip(jnp.exp(logT), cfg.T_min, cfg.T_max)

        new_state = ChEESState(key, q1_new, q2_new, mu_new, T_new)
        stats = jnp.mean(a_concat)
        return new_state, stats

    # Warmup (adapt T)
    state, accept_hist_warm = jax.lax.scan(one_step_warmup, state, xs=None, length=cfg.n_warmup)

    # Freeze T, collect samples (no ChEES gradient here)
    def one_step_sample(state: ChEESState, _):
        key = state.key
        key, k1, k2, ka1, ka2 = jax.random.split(key, 5)

        q1_prop, _, _, _, a_prob1, _ = hmc_walk_move(k1, state.q1, state.q2, logprob_and_grad, cfg.eps, state.T)
        q2_prop, _, _, _, a_prob2, _ = hmc_walk_move(k2, state.q2, state.q1, logprob_and_grad, cfg.eps, state.T)

        q1_new = accept_proposal(ka1, state.q1, q1_prop, a_prob1)
        q2_new = accept_proposal(ka2, state.q2, q2_prop, a_prob2)

        q_concat = jnp.concatenate([q1_new, q2_new], axis=0)
        mu_new = welford_ema_mean(state.mu, q_concat, cfg.mu_momentum)
        new_state = ChEESState(key, q1_new, q2_new, mu_new, state.T)
        a_mean = jnp.mean(jnp.concatenate([a_prob1, a_prob2], axis=0))
        return new_state, (q_concat, a_mean)

    state, outs = jax.lax.scan(one_step_sample, state, xs=None, length=cfg.n_samples)
    samples, accept_hist_samp = outs

    accept_hist = jnp.concatenate([accept_hist_warm, accept_hist_samp], axis=0)
    return state, samples, accept_hist


# ----------------------- Example target -----------------------

def gaussian_logprob_and_grad(mean: Array, cov: Array) -> LogProbFn:
    """Return a batched logprob_and_grad for N(mean, cov) with fixed cov.
    Args:
        mean: (d,)
        cov: (d,d) SPD
    """

    def fn(q: Array) -> Tuple[Array, Array]:
        centered = q - mean  # (B, d)
        # Gaussian log-density (up to const): -0.5 * x^T C x, with C = cov (d,d)
        # Use matrix multiplication instead of einsum for clarity
        quad = jnp.sum(centered @ cov * centered, axis=-1)  # (B,)
        logp = -0.5 * quad  # (B,)
        # Gradient of logp wrt q: - C x
        grad_logp = - (centered @ cov)  # (B, d)
        # Return potential U = -logp and its gradient
        return -logp, -grad_logp

    return fn


# ----------------------- Demo -----------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    d = 5
    # Make an ill-conditioned Gaussian to test path-length tuning
    cond_number = 10000
    eigenvals = 0.1 * np.linspace(1, cond_number, d)
    H = np.random.randn(d, d)
    Q, _ = np.linalg.qr(H)
    precision = Q @ np.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)
    
    cov = jnp.array(precision)
    mean = jnp.zeros((d,))

    target = gaussian_logprob_and_grad(mean, cov)

    # --- Euclidean ChEES ---
    cfg_euc = ChEESConfig(n_chains=64, n_warmup=1500, n_samples=1000, dim=d, eps=0.05, T_init=1.0,
                          chees_metric="euclidean")
    state_euc, samples_euc, accept_hist_euc = run_chees(key, target, cfg_euc)
    print("[Euclidean] Final T:", float(state_euc.T))
    corner.corner(np.array(samples_euc).reshape(-1, d))
    plt.show()

    # --- Metric-aware ChEES using current chains as proxy (for demo) ---
    # In an ensemble implementation, pass positions_for_metric that returns the
    # *complement group* positions for true affine invariance.
    cfg_aff = ChEESConfig(n_chains=10, n_warmup=1500, n_samples=1000, dim=d, eps=0.05, T_init=1.0,
                          chees_metric="mahalanobis", metric_lam=1e-6,
                          positions_for_metric=lambda st: jnp.concatenate([st.q1, st.q2], axis=0))
    state_aff, samples_aff, accept_hist_aff = run_chees(key, target, cfg_aff)
    print("[Affine/ChEESΣ] Final T:", float(state_aff.T))

    corner.corner(np.array(samples_aff).reshape(-1, d))
    plt.show()
