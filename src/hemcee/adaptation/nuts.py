'''
JAX implementation of iterative build-tree (Algorithm 2 in https://arxiv.org/pdf/1912.11554)
'''
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------
# Problem-specific state container (customize as you like)
# ---------------------------------------------------------------------
class Tree(NamedTuple):
    '''
    Args:
        left (State): The leftmost state stored at S[0]
        right (State): The rightmost state
        stop (bool): Whether the tree has stopped
    '''
    left: any
    right: any
    stop: jnp.bool_

# ---------------------------------------------------------------------
# Helpers: BitCount and TrailingBit (as used in the paper/pseudocode)
# ---------------------------------------------------------------------
def _as_unsigned(x):
    x = jnp.asarray(x)
    if x.dtype == jnp.int32:
        return x.astype(jnp.uint32)
    if x.dtype == jnp.int64:
        return x.astype(jnp.uint64)
    raise TypeError(f"Unsupported dtype {x.dtype}; expected int32 or int64.")

def bit_count(x):
    xu = _as_unsigned(x)
    return jax.lax.population_count(xu).astype(x.dtype)

def trailing_ones(x):
    # identity: trailing_ones(x) = popcount(x ^ (x + 1)) - 1
    xu = _as_unsigned(x)
    one = jnp.array(1, dtype=xu.dtype)
    return (jax.lax.population_count(xu ^ (xu + one)) - one).astype(x.dtype)

# ---------------------------------------------------------------------
# IterativeBuildTree (Algorithm 2)
# ---------------------------------------------------------------------
# ====== PyTree array helpers for level storage S[0..d-1] ======
def _replicate_state(z, d: int):
    """Make a length-d stack (axis 0) filled with copies of state z."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (d,) + x.shape), z)

def _take_state(S, i: jnp.int32):
    """S[i] with PyTree indexing."""
    return jax.tree.map(lambda a: jax.lax.dynamic_index_in_dim(a, i, keepdims=False), S)

def _set_state(S, i: jnp.int32, z):
    """Return S with S[i] = z (PyTree)."""
    return jax.tree.map(
        lambda a, v: jax.lax.dynamic_update_index_in_dim(a, v[None, ...], i, axis=0),
        S, z
    )


# ====== Main iterative build-tree with lax control flow ======
def build_tree(
    z0,
    d: int,
    *,
    is_u_turn_fn: Callable[[any, any], jnp.bool_],
    leapfrog_fn: Callable[[any], any],
):
    """
    Pure-JAX (lax.*) iterative build-tree (NUTS-like).
    Creates 2^d leapfrog steps unless an early U-turn is detected.

    Args:
        z0: initial State (PyTree)
        d : depth (non-negative int)
        is_u_turn_fn: (state_left, state_right) -> bool (JAX-friendly)
        leapfrog_fn : state -> state (JAX-friendly)

    Returns:
        Tree(left=S[0] (or z0 if d==0), right=z, stop=bool)
    """
    if d < 0:
        raise ValueError("Depth d must be >= 0")

    D = int(max(d, 1))  # storage length (avoid zero-length arrays)
    num_steps = jnp.int32(1 << d)  # 2^d

    # Storage S[0..d-1] (unused if d == 0)
    S0 = _replicate_state(z0, D)

    class Carry(NamedTuple):
        z: any
        S: any
        stop: jnp.bool_

    carry0 = Carry(z=z0, S=S0, stop=jnp.bool_(False))

    def body_fun(n: jnp.int32, carry: Carry):
        # If we've already stopped, don't advance anything further.
        def when_stopped(_):
            return carry

        def when_running(_):
            z, S, stop = carry

            # z <- LEAPFROG(z)
            z1 = leapfrog_fn(z)

            is_even = (n & jnp.int32(1)) == jnp.int32(0)

            # --- Even n branch: store S[i] <- z1 at i = bit_count(n) if 0<=i<d ---
            def even_branch(_):
                i = bit_count(n)
                def store_if_needed(S_in):
                    cond = (d > 0) & (i >= 0) & (i < jnp.int32(d))
                    S_out = jax.lax.cond(
                        cond,
                        lambda _S: _set_state(_S, i, z1),
                        lambda _S: _S,
                        S_in
                    )
                    return S_out
                S2 = store_if_needed(S)
                return Carry(z=z1, S=S2, stop=stop)

            # --- Odd n branch: check U-turns over k = i_max, ..., i_min ---
            def odd_branch(_):
                # l = trailing ones(n)
                l = trailing_ones(n)
                i_max = bit_count(n - jnp.int32(1))
                i_min = i_max - l + jnp.int32(1)

                # We'll scan k downwards via while_loop, carrying stop flag and leftmost state
                class Inner(NamedTuple):
                    k: jnp.int32
                    stop: jnp.bool_
                    left0: any   # S[0] (for return); keep it around
                    S: any
                    z_cur: any

                inner0 = Inner(
                    k=i_max,
                    stop=jnp.bool_(False),
                    left0=_take_state(S, jnp.int32(0)),
                    S=S,
                    z_cur=z1
                )

                def cond_fun(inner: Inner):
                    return (~inner.stop) & (inner.k >= i_min)

                def body_fun_inner(inner: Inner):
                    # Only check when 0 <= k < d
                    in_range = (inner.k >= 0) & (inner.k < jnp.int32(d))
                    left_k = _take_state(inner.S, jnp.clip(inner.k, 0, jnp.int32(D-1)))
                    # If out of range, we won't check; define turning=False in that case
                    turning = jax.lax.cond(
                        in_range,
                        lambda _: is_u_turn_fn(left_k, inner.z_cur),
                        lambda _: jnp.bool_(False),
                        operand=None
                    )
                    stop2 = inner.stop | turning
                    # decrement k
                    return Inner(k=inner.k - jnp.int32(1),
                                 stop=stop2,
                                 left0=inner.left0,
                                 S=inner.S,
                                 z_cur=inner.z_cur)

                inner_final = jax.lax.while_loop(cond_fun, body_fun_inner, inner0)

                return Carry(z=z1, S=inner_final.S, stop=inner_final.stop)

            new_carry = jax.lax.cond(is_even, even_branch, odd_branch, operand=None)
            return new_carry

        return jax.lax.cond(carry.stop, when_stopped, when_running, operand=None)

    carry_final = jax.lax.fori_loop(jnp.int32(0), num_steps, body_fun, carry0)

    # Leftmost = S[0] if d>0 else z0
    leftmost = jax.lax.cond(
        d > 0,
        lambda S: _take_state(S, jnp.int32(0)),
        lambda _: z0,
        carry_final.S
    )
    return Tree(left=leftmost, right=carry_final.z, stop=carry_final.stop)