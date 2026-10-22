"""``std::mt19937`` in JAX, so a level seed means the same level as the original.

ProcGen seeds one ``RandGen`` per episode (``rand_gen.seed(current_level_seed)``)
and draws with ``x % n``; both are fully specified, so reproducing them bit for
bit is a matter of implementing the generator, not of matching an RNG library.
:class:`Mt19937` carries the 624-word state and the cursor as a pytree, which
keeps it usable inside ``lax.scan``/``lax.cond`` and therefore inside jitted
level generation.

Draw counts per level are in the tens, far below the 624 outputs in a block, so
the twist only runs when a caller asks for more than that; it is still
implemented (sequentially, like the C++ in-place loop) so the generator can be
used for anything.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

N = 624
M = 397
MATRIX_A = 0x9908B0DF
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7FFFFFFF
#: 2**32, the denominator of ``rand01``.
TWO_POW_32 = 4294967296.0


class Mt19937(NamedTuple):
    """The generator state: 624 words, a cursor, and the seeding flag."""

    words: jax.Array  # (624,) uint32
    index: jax.Array  # () int32, 624 means "twist before the next draw"
    seeded: jax.Array  # () bool


def seed(value: jax.Array) -> Mt19937:
    """``std::mt19937::seed``: the standard 1812433253 recurrence."""
    value = jnp.asarray(value, jnp.uint32)
    words = jnp.zeros((N,), jnp.uint32).at[0].set(value)

    def body(index, words):
        previous = words[index - 1]
        mixed = jnp.uint32(1812433253) * (previous ^ (previous >> jnp.uint32(30)))
        return words.at[index].set(mixed + jnp.uint32(index))

    words = jax.lax.fori_loop(1, N, body, words)
    return Mt19937(words=words, index=jnp.asarray(N, jnp.int32), seeded=jnp.asarray(True))


def _twist(words: jax.Array) -> jax.Array:
    """Regenerate the block in place, exactly like the reference implementation."""

    def body(i, words):
        y = (words[i] & jnp.uint32(UPPER_MASK)) | (words[(i + 1) % N] & jnp.uint32(LOWER_MASK))
        mixed = words[(i + M) % N] ^ (y >> jnp.uint32(1))
        mixed = jnp.where(y & jnp.uint32(1) != 0, mixed ^ jnp.uint32(MATRIX_A), mixed)
        return words.at[i].set(mixed)

    return jax.lax.fori_loop(0, N, body, words)


def next_uint32(state: Mt19937):
    """One raw 32-bit output, with tempering."""
    words = jax.lax.cond(state.index >= N, lambda s: _twist(s), lambda s: s, state.words)
    index = jnp.where(state.index >= N, 0, state.index)
    y = words[index]
    y = y ^ (y >> jnp.uint32(11))
    y = y ^ ((y << jnp.uint32(7)) & jnp.uint32(0x9D2C5680))
    y = y ^ ((y << jnp.uint32(15)) & jnp.uint32(0xEFC60000))
    y = y ^ (y >> jnp.uint32(18))
    return state._replace(words=words, index=index + 1), y


def randn(state: Mt19937, high):
    """``RandGen::randn``: a uniform integer in ``[0, high)``."""
    state, x = next_uint32(state)
    return state, (x % jnp.asarray(high, jnp.uint32)).astype(jnp.int32)


def randint(state: Mt19937, low, high):
    """``RandGen::randint``: a uniform integer in ``[low, high)``."""
    state, x = next_uint32(state)
    span = jnp.asarray(high - low, jnp.uint32)
    return state, (x % span).astype(jnp.int32) + low


def rand01(state: Mt19937):
    """``RandGen::rand01``: ``x / 2**32`` as float32."""
    state, x = next_uint32(state)
    return state, x.astype(jnp.float32) / TWO_POW_32
