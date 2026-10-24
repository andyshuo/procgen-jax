"""Parity of the room generator with ProcGen's ``roomgen.cpp``.

``tools/build_roomgen_reference.py`` compiles the original against a grid-only
stand-in for ``BasicAbstractGame`` and records, for five jumper-shaped grids, the
cellular-automata pass, the best room, an expanded room and a path.  Everything here
has to match exactly: two quirks (the room's seed cell is left out, the path's
``covered`` set starts empty) decide the levels jumper builds.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import roomgen

FIXTURE = Path(__file__).parent / 'data' / 'roomgen_reference.npz'


@pytest.fixture(scope='module')
def reference():
    if not FIXTURE.exists():
        pytest.skip('run `pixi run roomgen-reference` to generate the C++ fixture')
    return np.load(FIXTURE)


def test_the_cellular_automata_pass_matches(reference):
    for index in range(len(reference['seeds'])):
        flat = jnp.asarray(reference['init'][index].reshape(-1), jnp.int32)

        # the reference runs two passes, which is what jumper does
        out = roomgen.update(roomgen.update(flat, 40), 40)

        assert np.array_equal(np.asarray(out), reference['ca'][index].reshape(-1))


def test_the_best_room_matches(reference):
    for index in range(len(reference['seeds'])):
        flat = jnp.asarray(reference['ca'][index].reshape(-1), jnp.int32)

        mask = roomgen.find_best_room(flat, 40)

        assert np.array_equal(np.asarray(mask), reference['best'][index].reshape(-1)), f'case {index}: rooms differ'


def test_expanding_the_room_matches(reference):
    for index in range(len(reference['seeds'])):
        flat = jnp.asarray(reference['ca'][index].reshape(-1), jnp.int32)
        room = jnp.asarray(reference['best'][index].reshape(-1))

        grown = roomgen.expand_room(flat, 40, room, 3)

        assert np.array_equal(np.asarray(grown), reference['expanded'][index].reshape(-1))


def test_the_path_matches(reference):
    for index in range(len(reference['seeds'])):
        flat = jnp.asarray(reference['ca'][index].reshape(-1), jnp.int32)
        component = np.flatnonzero(reference['best'][index].reshape(-1))
        src = jnp.asarray(int(component.min()), jnp.int32)
        dst = jnp.asarray(int(component.max()), jnp.int32)

        cells, length = roomgen.find_path(flat, 40, src, dst)

        length = int(length)
        expected = reference['paths'][index][: reference['path_lengths'][index]]
        assert length == int(reference['path_lengths'][index]), f'case {index}: path length'
        assert np.array_equal(np.asarray(cells)[:length], expected), f'case {index}: path cells'


def test_build_room_returns_the_whole_component(reference):
    """The seed is not inserted up front, but it is rediscovered from its neighbours."""
    flat = jnp.asarray(reference['ca'][0].reshape(-1), jnp.int32)
    expected = np.asarray(roomgen.find_best_room(flat, 40))
    seed = jnp.asarray(int(np.argmax(expected)), jnp.int32)

    room = np.asarray(roomgen.build_room(flat, 40, seed))

    assert bool(room[int(seed)]), 'the seed comes back as a neighbour'
    assert np.array_equal(room, expected)


def test_an_isolated_cell_has_no_room():
    """A one-cell component is the one case the seed really is missing from."""
    flat = jnp.full((16,), roomgen.WALL_OBJ, jnp.int32).at[5].set(roomgen.SPACE)

    room = roomgen.build_room(flat, 4, jnp.asarray(5, jnp.int32))

    assert not bool(room.any())


def test_a_room_is_connected_and_open(reference):
    flat = np.asarray(reference['ca'][0].reshape(-1))
    mask = np.asarray(roomgen.find_best_room(jnp.asarray(flat, jnp.int32), 40))

    assert mask.sum() > 100
    assert np.all(flat[mask] == roomgen.SPACE), 'a room only holds open cells'


def test_the_generator_is_traceable(reference):
    traced = jax.jit(roomgen.update, static_argnums=(1,))
    flat = jnp.asarray(reference['init'][3].reshape(-1), jnp.int32)

    out = traced(traced(flat, 40), 40)

    assert np.array_equal(np.asarray(out), reference['ca'][3].reshape(-1))
