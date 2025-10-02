import jax
import jax.numpy as jnp
import functools
from typing import Callable, NamedTuple

from hemcee.adaptation.nuts import build_tree as build_tree_lax

"""
Implementation of iterative build-tree (Algorithm 2 in https://arxiv.org/pdf/1912.11554)
"""

# ---------- State / Tree ----------
class State(NamedTuple):
    q: jnp.ndarray
    p: jnp.ndarray

class Tree(NamedTuple):
    left: any
    right: any
    stop: jnp.bool_

# ---------- Build tree, original implementation ----------
def build_tree_original(
    z0: State,
    d: int,
    *,
    is_u_turn_fn: Callable[[State, State], bool],
    leapfrog_fn: Callable[[State], State],
) -> Tree:
    if d < 0:
        raise ValueError("Depth d must be >= 0")
    S = [z0] * max(d, 1)
    z = z0
    num_steps = 1 << d
    
    def bit_count(n): 
        return int(bin(n).count("1"))
    
    def trailing_ones(n):
        # number of consecutive 1-bits at LSB: ctz(n+1)
        x = n + 1
        c = 0
        while x & 1 == 0:
            c += 1
            x >>= 1
        return c

    for n in range(num_steps):
        z = leapfrog_fn(z)
        if (n % 2) == 0:
            i = bit_count(n)
            if d > 0 and 0 <= i < d:
                S[i] = z
        else:
            l = trailing_ones(n)
            i_max = bit_count(n - 1)
            i_min = i_max - l + 1
            for k in range(i_max, i_min - 1, -1):
                if 0 <= k < d:
                    if is_u_turn_fn(S[k], z):
                        return Tree(S[0] if d > 0 else z0, z, True)
    leftmost = S[0] if d > 0 else z0
    return Tree(leftmost, z, False)

# ---------- Simple harmonic oscillator test system ----------
def make_leapfrog(dt: float):
    """Velocity Verlet / leapfrog for U(q)=0.5*||q||^2, M=I."""
    dt = float(dt)
    dt = jnp.asarray(dt)

    def grad_U(q):  # ∇U = q
        return q

    def lf(z: State) -> State:
        q, p = z.q, z.p
        p_half = p - 0.5 * dt * grad_U(q)
        q_new = q + dt * p_half
        p_new = p_half - 0.5 * dt * grad_U(q_new)
        return State(q_new, p_new)

    return lf

def nuts_uturn(s_left: State, s_right: State) -> jnp.bool_:
    """
    Standard NUTS check for a one-sided edge: (θ_right - θ_left)·p_right ≤ 0
    (You can swap in a symmetric check if you prefer.)
    """
    dq = s_right.q - s_left.q
    return (jnp.dot(dq, s_right.p) <= 0.0)

# ---------- Tests ----------
def _allclose_state(a: State, b: State, tol=1e-6):
    return (jnp.allclose(a.q, b.q, atol=tol, rtol=tol) &
            jnp.allclose(a.p, b.p, atol=tol, rtol=tol))

def test_equivalence_no_stop():
    """Predicate always False → both versions should do 2^d steps and agree."""
    d = 4
    z0 = State(q=jnp.array([1.0, -2.0]), p=jnp.array([0.3, 0.7]))
    leapfrog = make_leapfrog(dt=0.05)
    never_stop = lambda a, b: jnp.bool_(False)

    t_ref = build_tree_original(z0, d, is_u_turn_fn=lambda a,b: False, leapfrog_fn=leapfrog)
    t_jax = build_tree_lax(z0, d, is_u_turn_fn=never_stop, leapfrog_fn=leapfrog)

    assert not bool(t_ref.stop)
    assert not bool(t_jax.stop)
    assert _allclose_state(t_ref.left, t_jax.left)
    assert _allclose_state(t_ref.right, t_jax.right)

def test_equivalence_with_stop():
    """U-turn triggers early → both versions return same break state."""
    d = 10  # large enough that early-stop likely before 2^d
    z0 = State(q=jnp.array([2.0]), p=jnp.array([2.5]))  # energy high → quick turn
    leapfrog = make_leapfrog(dt=0.2)

    t_ref = build_tree_original(z0, d, is_u_turn_fn=lambda a,b: bool(nuts_uturn(a,b)),
                              leapfrog_fn=leapfrog)
    t_jax = build_tree_lax(z0, d, is_u_turn_fn=nuts_uturn, leapfrog_fn=leapfrog)

    assert bool(t_ref.stop)
    assert bool(t_jax.stop)
    assert _allclose_state(t_ref.left, t_jax.left)
    assert _allclose_state(t_ref.right, t_jax.right)

def test_leftmost_when_d0():
    """When d=0, left should be z0 and we still advance one step at n=0."""
    d = 0
    z0 = State(q=jnp.array([1.0]), p=jnp.array([0.0]))
    leapfrog = make_leapfrog(dt=0.1)
    never_stop = lambda a,b: jnp.bool_(False)

    t = build_tree_lax(z0, d, is_u_turn_fn=never_stop, leapfrog_fn=leapfrog)
    # left=z0, right=leapfrog(z0)
    assert jnp.allclose(t.left.q, z0.q) and jnp.allclose(t.left.p, z0.p)
    z1 = leapfrog(z0)
    assert _allclose_state(t.right, z1)
    assert not bool(t.stop)

def test_jit_compiles():
    """JIT should compile and run with static d."""
    leapfrog = make_leapfrog(0.05)
    kernel = jax.jit(
        functools.partial(build_tree_lax, d=5, is_u_turn_fn=nuts_uturn, leapfrog_fn=leapfrog),
        static_argnames=()  # d and fns already bound
    )
    z0 = State(q=jnp.array([0.7, -0.1]), p=jnp.array([0.2, 0.4]))
    out = kernel(z0)
    assert isinstance(out.stop, jnp.ndarray) or isinstance(out.stop, (bool, jnp.bool_))

def test_vmap_over_batch():
    """Vectorize over multiple initial states (chains)."""
    leapfrog = make_leapfrog(0.03)

    def run_single(z0: State):
        return build_tree_lax(z0, 4, is_u_turn_fn=nuts_uturn, leapfrog_fn=leapfrog)

    batched_run = jax.vmap(run_single)
    z0_batch = State(
        q=jnp.stack([jnp.array([1.0, 0.0]), jnp.array([0.5, -0.3])], axis=0),
        p=jnp.stack([jnp.array([0.1, 0.2]), jnp.array([0.7, 0.4])], axis=0),
    )
    tree_batch = batched_run(z0_batch)
    # spot-check shapes
    assert tree_batch.left.q.shape == (2, 2)
    assert tree_batch.right.q.shape == (2, 2)

# --------- Run tests (if executing as a script / notebook) ----------

if __name__ == "__main__":
    test_equivalence_no_stop()
    test_equivalence_with_stop()
    test_leftmost_when_d0()
    test_jit_compiles()
    test_vmap_over_batch()
    print("All tests passed ✅")