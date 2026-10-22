"""MT19937 taken from ProcGen's helper, checked against the reference values.

The expected numbers below were produced by compiling ``procgen/src/randgen.cpp``
itself (see ``tools/build_reference.py``), not by another RNG library, so this
pins the exact generator the levels are built from.
"""

import jax
import jax.numpy as jnp
import pytest

from procgen_jax import rng

#: ``RandGen`` seeded with 5489, six raw draws.
RAW_5489 = [3499211612, 581869302, 3890346734, 3586334585, 545404204, 4161255391]
#: ``RandGen`` seeded with 5489: ``randn(3)`` eight times.
RANDN3_5489 = [2, 0, 2, 2, 1, 1, 2, 2]
#: ``RandGen`` seeded with 0: ``randn(4)`` six times.
RANDN4_0 = [0, 3, 1, 0, 3, 3]
#: ``RandGen`` seeded with 0: ``rand01()``.
RAND01_0 = 0.548813522


def test_raw_output_matches_the_reference_generator():
    state = rng.seed(jnp.asarray(5489))

    draws = []
    for _ in range(len(RAW_5489)):
        state, value = rng.next_uint32(state)
        draws.append(int(value))

    assert draws == RAW_5489


def test_randn_matches_the_reference_generator():
    state = rng.seed(jnp.asarray(5489))
    draws = []
    for _ in range(len(RANDN3_5489)):
        state, value = rng.randn(state, 3)
        draws.append(int(value))
    assert draws == RANDN3_5489

    state = rng.seed(jnp.asarray(0))
    draws = []
    for _ in range(len(RANDN4_0)):
        state, value = rng.randn(state, 4)
        draws.append(int(value))
    assert draws == RANDN4_0


def test_rand01_matches_the_reference_generator():
    state = rng.seed(jnp.asarray(0))

    state, value = rng.rand01(state)

    assert abs(float(value) - RAND01_0) < 1e-6


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_seeding_is_reproducible(seed):
    first = rng.seed(jnp.asarray(seed))
    second = rng.seed(jnp.asarray(seed))

    for _ in range(5):
        first, a = rng.next_uint32(first)
        second, b = rng.next_uint32(second)
        assert int(a) == int(b)


def test_a_block_of_624_draws_is_unique():
    """The state must twist once the 624-word block is exhausted."""
    state = rng.seed(jnp.asarray(1234))

    values = []
    for _ in range(700):
        state, value = rng.next_uint32(state)
        values.append(int(value))

    assert len(set(values)) == len(values)


def test_state_is_a_pytree():
    """The generator has to survive ``jit``, so its state must flatten."""
    state = rng.seed(jnp.asarray(7))

    @jax.jit
    def draw(state):
        state, value = rng.next_uint32(state)
        return state, value

    state, value = draw(state)

    assert jnp.isfinite(value.astype(jnp.float32))
    assert state.words.shape == (rng.N,)
