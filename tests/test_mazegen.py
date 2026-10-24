"""Parity of the shared maze generator with ProcGen's ``mazegen.cpp``.

``tools/build_maze_reference.py`` compiles ProcGen's own generator against its own
``RandGen`` and records 35 grids in ``tests/data/maze_reference.npz``.  Every case
has to match cell for cell: a divergence means the wall-removal order, the union
bookkeeping or a draw was transcribed wrongly.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import mazegen, rng
from procgen_jax.engine import WALL_OBJ
from procgen_jax.objects import PROCGEN_SPACE as SPACE

FIXTURE = Path(__file__).parent / 'data' / 'maze_reference.npz'

#: Every mode this port implements so far (``with_doors`` is still to come).
MODES = {
    'maze': mazegen.generate_maze,
    'no_dead_ends': mazegen.generate_maze_no_dead_ends,
}


@pytest.fixture(scope='module')
def reference():
    if not FIXTURE.exists():
        pytest.skip('run `pixi run maze-reference` to generate the C++ fixture')
    return np.load(FIXTURE)


def build(mode: str, seed: int, maze_dim: int):
    state = rng.seed(jnp.asarray(seed, jnp.uint32))
    state, grid = MODES[mode](state, maze_dim)
    return np.asarray(grid)


def test_every_generated_maze_matches_the_reference(reference):
    checked = 0
    for index in range(len(reference['modes'])):
        mode = str(reference['modes'][index])
        if mode not in MODES:
            continue
        seed = int(reference['seeds'][index])
        maze_dim = int(reference['dims'][index])
        expected = reference['grids'][index][: maze_dim + 2, : maze_dim + 2]

        grid = build(mode, seed, maze_dim)

        assert np.array_equal(grid, expected), f'{mode} seed {seed} dim {maze_dim} diverges'
        checked += 1

    assert checked >= 20, 'the fixture should cover both modes and both sizes'


def test_the_grid_only_holds_walls_and_space(reference):
    grid = build('maze', 7, 13)

    assert set(np.unique(grid)) <= {WALL_OBJ, SPACE, 0}


def test_the_border_is_solid(reference):
    grid = build('maze', 3, 11)

    assert np.all(grid[0] == WALL_OBJ) and np.all(grid[-1] == WALL_OBJ)
    assert np.all(grid[:, 0] == WALL_OBJ) and np.all(grid[:, -1] == WALL_OBJ)


def test_cells_are_connected():
    """A Kruskal maze is spanning: every open cell is reachable from every other."""
    from collections import deque

    grid = build('maze', 5, 13)
    start = tuple(np.argwhere(grid == SPACE)[0])
    seen, queue = {start}, deque([start])
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = (y + dy, x + dx)
            if cell in seen:
                continue
            if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
                continue
            if grid[cell] == WALL_OBJ:
                continue
            seen.add(cell)
            queue.append(cell)

    open_cells = set(map(tuple, np.argwhere(grid == SPACE)))
    assert seen == open_cells, 'the maze must not have isolated pockets'


def test_no_dead_ends_removes_dead_ends():
    """The variant punches at least as many openings as the plain maze."""
    plain = build('maze', 11, 13)
    carved = build('no_dead_ends', 11, 13)

    assert (carved == SPACE).sum() >= (plain == SPACE).sum()


def test_generation_is_deterministic():
    assert np.array_equal(build('maze', 42, 9), build('maze', 42, 9))


def test_different_seeds_give_different_mazes():
    assert not np.array_equal(build('maze', 1, 9), build('maze', 2, 9))


def test_wall_list_matches_the_original_layout():
    """Walls are enumerated odd-row first, then odd-column, as in the C++."""
    walls = mazegen.wall_list(9)

    assert len(walls) == 40
    assert walls[0] == (0, 0, 2, 0)
    assert walls[4] == (0, 8, 2, 8), 'first row of the second wall column'
    assert walls[20] == (0, 0, 0, 2), 'the vertical walls start after the horizontal ones'
