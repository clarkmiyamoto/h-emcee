import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from hemcee.adaptation.nuts import bit_count, trailing_ones

test_vals = jnp.array([
    0b0,
    0b1,
    0b10,
    0b1011,       # 11
    0b10111,      # 23
    0b1111,       # 15
    0b1000,       # 8
    0b1111000,    # 120
    (1 << 31) - 1,  # max int32 (all ones)
    (1 << 63) - 1,  # max int64 (all ones)
    (1 << 32) - 1,  # 0xFFFFFFFF  -> will wrap to -1 in int32
], dtype=jnp.int64)

correct_bit_count = jnp.array([
    0,
    1,
    1,
    3,
    4,
    4,
    1,
    4,
    31,
    63,
    32,
], dtype=jnp.int64)

correct_trailing_ones = jnp.array([
    0,
    1,
    0,
    2,
    3,
    4,
    0,
    0,
    31,
    63,
    32,
], dtype=jnp.int64)

def test_bit_count():
    assert jnp.array_equal(bit_count(test_vals), correct_bit_count)

def test_trailing_ones():
    assert jnp.array_equal(trailing_ones(test_vals), correct_trailing_ones)