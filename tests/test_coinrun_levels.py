"""Level parity with the original C++ implementation.

``tools/build_reference.py`` compiles ProcGen's own ``generate_coin_to_the_right``
against its own ``RandGen`` and stores the resulting grids in
``tests/data/coinrun_levels.npz``.  The tiles and the entity list have to match
exactly: a divergence means the draw order or a branch differs from the original.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax.games.coinrun import generate_level as _generate_level

# Level generation is a scan of masked grid fills; eager it costs seconds per
# call, so the tests drive the jitted version.
generate_level = jax.jit(_generate_level)

FIXTURE = Path(__file__).parent / 'data' / 'coinrun_levels.npz'
#: The original stores empty tiles as ``SPACE == 100``; this port uses 0.
PROCGEN_SPACE = 100


@pytest.fixture(scope='module')
def reference():
    if not FIXTURE.exists():
        pytest.skip('run `pixi run reference` to generate the C++ fixture')
    return np.load(FIXTURE)


def entities_of(level) -> list[tuple[int, float, float, float, int]]:
    """Every body of a generated level, as ``(type, x, y, vx, theme)``."""
    _, enemies, statics, num_enemies, num_statics, _, _ = level
    bodies = []
    for pool, count in ((enemies, int(num_enemies)), (statics, int(num_statics))):
        for index in range(count):
            bodies.append(
                (
                    int(pool.type[index]),
                    round(float(pool.x[index]), 4),
                    round(float(pool.y[index]), 4),
                    round(float(pool.vx[index]), 4),
                    int(pool.image_theme[index]),
                )
            )
    return sorted(bodies)


def test_level_grids_are_identical_to_the_reference(reference):
    for index, seed in enumerate(reference['seeds']):
        grid = np.asarray(generate_level(jnp.asarray(int(seed), jnp.int32))[0])
        expected = np.where(reference['grids'][index] == PROCGEN_SPACE, 0, reference['grids'][index])

        difference = int((grid != expected).sum())
        assert difference == 0, f'seed {int(seed)}: {difference} tiles differ from the C++ level'


def test_entity_lists_are_identical_to_the_reference(reference):
    for index, seed in enumerate(reference['seeds']):
        level = generate_level(jnp.asarray(int(seed), jnp.int32))
        expected = reference['entities'][index][: int(reference['counts'][index])]
        expected_bodies = sorted(
            (
                int(kind),
                round(float(x), 4),
                round(float(y), 4),
                round(float(vx), 4),
                int(theme),
            )
            for kind, x, y, vx, theme in expected
        )

        assert entities_of(level) == expected_bodies, f'seed {int(seed)}: entity lists differ'


def test_the_ground_theme_and_player_colour_come_from_the_same_stream(reference):
    """Both are drawn before the level, so they shift every later draw."""
    themes = set()
    for seed in range(12):
        _, _, _, _, _, wall_theme, player_colour = generate_level(jnp.asarray(seed, jnp.int32))
        themes.add((int(wall_theme), int(player_colour)))

    assert len(themes) > 1, 'theme selection should vary across seeds'
