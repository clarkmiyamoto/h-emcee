"""
ChEES-HMC (Euclidean metric) in JAX

Implements the original ChEES adaptation of the integration time T for HMC
with a fixed stepsize epsilon. The objective is eq. (7) in Hoffman–Radul–Sountsov
(ChEES): maximize E[ 1/4 (||q' - mu||^2 - ||q - mu||^2)^2 ].

This code:
- Runs N chains in parallel (vmap) with a shared epsilon and adaptive T
- Uses jittered integration time t = u * T with u ~ Uniform(0,1)
- Uses leapfrog integrator with L = max(1, round(t / epsilon)) steps
- Computes a per-iteration stochastic gradient estimator for d/dT of the ChEES
  criterion and performs gradient-ascent on log T (bounded)
- Maintains a running mean mu of positions for the criterion (Welford EMA)

Notes:
- This is the *original* (Euclidean) ChEES. A metric-aware variant simply replaces
  Euclidean inner products/norms with Mahalanobis via a user-provided metric.
- For stability, we (i) damp the ChEES gradient by acceptance indicator and
  (ii) use an EMA for mu with momentum `mu_momentum`.

Author: ChatGPT (JAX)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
LogProbFn = Callable[[Array], Tuple[Array, Array]]  # returns (logp, grad)


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


# ----------------------- Leapfrog -----------------------

def leapfrog(q: Array, p: Array, grad_logprob: Callable[[Array], Array], eps: float, L: int) -> Tuple[Array, Array]:
   """Standard leapfrog integrator for HMC with Euclidean metric.
   Args:
      q: (B, d) positions
      p: (B, d) momenta
      grad_logprob: maps (B, d)->(B, d), gradient of log density wrt q
      eps: step size
      L: number of leapfrog steps (>=1)
   Returns:
      (q_new, p_new) after L steps
   """
   def body(carry, _):
      q, p = carry
      p_half = p + 0.5 * eps * grad_logprob(q)
      q_new = q + eps * p_half
      p_new = p_half + 0.5 * eps * grad_logprob(q_new)
      return (q_new, p_new), None

   (q_L, p_L), _ = jax.lax.scan(body, (q, p), xs=None, length=L)
   return q_L, p_L


# ----------------------- HMC step -----------------------

def hmc_propose(
    key: Array,
    q: Array,
    logprob_and_grad: LogProbFn,
    eps: float,
    T: float,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
   """One HMC proposal using jittered integration time t = u*T.
   Args:
      key: PRNG key
      q: (B, d) current positions
      logprob_and_grad: function returning (logp, grad) for a batch of q
      eps: step size
      T: max integration time
   Returns:
      q_prop, p_prop, q0, p0, accept_prob, t
   """
   B, d = q.shape

   key_u, key_p = jax.random.split(key)
   u = jax.random.uniform(key_u, shape=(B, 1))  # per-chain jitter
   t = u * T  # (B,1)
   L = jnp.maximum(1, jnp.round(t / eps).astype(int))  # (B,1)
   L = jnp.squeeze(L, axis=-1)

   # Sample momenta p ~ N(0, I)
   p0 = jax.random.normal(key_p, shape=q.shape)

   # Define batched grad
   def grad_logprob(x: Array) -> Array:
      _, g = logprob_and_grad(x)
      return g

   # Run leapfrog with per-chain L via while_loop (scan needs static length).
   # We'll loop each chain independently via vmap and a per-chain function.
   def per_chain_step(inputs):
      q_i, p_i, L_i = inputs

      def cond_fun(carry):
         q_c, p_c, i = carry
         return i < L_i
   
      def body_fun(carry):
         q_c, p_c, i = carry
         p_half = p_c + 0.5 * eps * grad_logprob(q_c[jnp.newaxis, :])[0]
         q_new  = q_c + eps * p_half
         p_new  = p_half + 0.5 * eps * grad_logprob(q_new[jnp.newaxis, :])[0]
         return (q_new, p_new, i + 1)
   
      q_Li, p_Li, _ = jax.lax.while_loop(
         cond_fun, body_fun, (q_i, p_i, jnp.array(0, dtype=L_i.dtype))
      )
      return q_Li, p_Li

   q_prop, p_prop = jax.vmap(per_chain_step)((q, p0, L))

   # MH accept prob
   logp0, _ = logprob_and_grad(q)
   logp_prop, _ = logprob_and_grad(q_prop)

   def kinetic(pp):
      return 0.5 * jnp.sum(pp * pp, axis=-1)  # (B,)

   H0 = -logp0 + kinetic(p0)
   H1 = -logp_prop + kinetic(p_prop)
   dH = H1 - H0
   accept_prob = jnp.exp(-jnp.minimum(0.0, dH)) * jnp.exp(-jnp.maximum(0.0, dH))
   # Numerically stable: accept_prob = exp(-dH) if dH>0 else 1
   accept_prob = jnp.where(dH > 0, jnp.exp(-dH), jnp.ones_like(dH))

   return q_prop, p_prop, q, p0, accept_prob, jnp.squeeze(t, -1)


# ----------------------- ChEES gradient -----------------------

def chees_grad_T(
    q0: Array, q1: Array, p1: Array, t: Array, mu: Array, accept: Array
) -> Array:
    """Per-iteration stochastic gradient estimator for d/dT of ChEES objective.
    Args:
        q0: (B, d) start positions
        q1: (B, d) proposed end positions
        p1: (B, d) end momenta (after final half step)
        t:  (B,)  actual integration times used (u*T per chain)
        mu: (d,)   running mean used in criterion
        accept: (B,) accept indicators or probabilities
    Returns:
        scalar gradient estimate wrt T (averaged over batch)
    """
    dq0 = q0 - mu  # (B, d)
    dq1 = q1 - mu  # (B, d)
    qnorm_diff = jnp.sum(dq1 * dq1, axis=-1) - jnp.sum(dq0 * dq0, axis=-1)  # (B,)
    inner = jnp.sum(dq1 * p1, axis=-1)  # (B,)
    # Gradient estimator (matches whitened Euclidean formula): t * (Δ||·||^2) * <dq1, p1>
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


class ChEESState(NamedTuple):
    key: Array
    q: Array           # (B,d)
    mu: Array          # (d,)
    T: float


def run_chees(
    key: Array,
    logprob_and_grad: LogProbFn,
    cfg: ChEESConfig,
    q0: Array | None = None,
) -> Tuple[ChEESState, Array, Array]:
    """Run ChEES-adapted HMC and return samples.
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

    state = ChEESState(key, q, mu, T)

    def one_step_warmup(state: ChEESState, _):
        key = state.key
        key, k = jax.random.split(key)
        q_prop, p_prop, q0, p0, a_prob, t = hmc_propose(k, state.q, logprob_and_grad, cfg.eps, state.T)
        # Accept/reject
        key, ka = jax.random.split(key)
        u = jax.random.uniform(ka, shape=a_prob.shape)
        accept = (u < a_prob).astype(q_prop.dtype)
        q_new = accept[:, None] * q_prop + (1 - accept)[:, None] * state.q

        # Update mu (EMA over accepted-or-not positions)
        mu_new = welford_ema_mean(state.mu, q_new, cfg.mu_momentum)

        # ChEES gradient wrt T (use probs as dampers)
        gT = chees_grad_T(q0, q_prop, p_prop, t, state.mu, a_prob)
        # Gradient ascent on log T (keep T positive, clamp bounds)
        logT = jnp.log(state.T)
        logT = logT + cfg.lr_T * gT
        T_new = jnp.clip(jnp.exp(logT), cfg.T_min, cfg.T_max)

        new_state = ChEESState(key, q_new, mu_new, T_new)
        stats = jnp.mean(a_prob)  # mean accept prob
        return new_state, stats

    # Warmup (adapt T)
    state, accept_hist_warm = jax.lax.scan(one_step_warmup, state, xs=None, length=cfg.n_warmup)

    # Freeze T, collect samples
    def one_step_sample(state: ChEESState, _):
        key = state.key
        key, k = jax.random.split(key)
        q_prop, p_prop, q0, p0, a_prob, t = hmc_propose(k, state.q, logprob_and_grad, cfg.eps, state.T)
        key, ka = jax.random.split(key)
        u = jax.random.uniform(ka, shape=a_prob.shape)
        accept = (u < a_prob).astype(q_prop.dtype)
        q_new = accept[:, None] * q_prop + (1 - accept)[:, None] * state.q
        mu_new = welford_ema_mean(state.mu, q_new, cfg.mu_momentum)
        new_state = ChEESState(key, q_new, mu_new, state.T)
        return new_state, (q_new, jnp.mean(a_prob))

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
    L = jnp.linalg.cholesky(cov)
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(L.shape[0]), lower=True)
    P = Linv.T @ Linv  # = cov^{-1}

    def fn(q: Array) -> Tuple[Array, Array]:
        dq = q - mean
        quad = jnp.einsum('bi,ij,bj->b', dq, P, dq)
        logp = -0.5 * (dq.shape[-1] * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov)) + quad)
        grad = - (dq @ P)  # gradient wrt q of -0.5*quad
        return logp, grad

    return fn


# ----------------------- Demo -----------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    d = 4
    # Make an ill-conditioned Gaussian to test path-length tuning
    eig = jnp.linspace(0.5, 25.0, d)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (d, d)))
    cov = (Q * eig) @ Q.T
    mean = jnp.zeros((d,))

    target = gaussian_logprob_and_grad(mean, cov)

    cfg = ChEESConfig(n_chains=64, n_warmup=1500, n_samples=1000, dim=d, eps=0.05, T_init=1.0)

    state, samples, accept_hist = run_chees(key, target, cfg)

    print("Final T:", float(state.T))
    print("Mean accept (last 200 iters):", float(jnp.mean(accept_hist[-200:])))
    print("Sample mean (first chain):", samples[-1, 0].tolist())
