"""Level generation: shapes, determinism and maze connectivity."""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax.games.maze import MazeEnv, generate_maze
from procgen_jax.objects import GOAL, SPACE, WALL


def bfs_reachable(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """Flood fill over non-wall tiles, 4-connected."""
    height, width = grid.shape
    seen = {start}
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and grid[ny, nx] != WALL:
                seen.add((ny, nx))
                queue.append((ny, nx))
    return seen


@pytest.mark.parametrize('world_dim', [3, 7, 15])
def test_generate_maze_shapes(world_dim):
    grid = generate_maze(jax.random.PRNGKey(0), world_dim)

    assert grid.shape == (world_dim, world_dim)
    assert grid.dtype == jnp.int32
    assert set(np.unique(np.asarray(grid))) <= {WALL, SPACE}


@pytest.mark.parametrize('world_dim', [7, 15])
def test_maze_is_perfect(world_dim):
    """A perfect maze is fully connected with no isolated pockets."""
    grid = np.asarray(generate_maze(jax.random.PRNGKey(3), world_dim))

    reachable = bfs_reachable(grid, (0, 0))
    open_tiles = set(zip(*np.where(grid != WALL), strict=False))
    assert reachable == open_tiles


def test_every_cell_centre_is_carved():
    """Cells sit on even coordinates and must all be walkable."""
    grid = np.asarray(generate_maze(jax.random.PRNGKey(1), 15))

    cells = grid[::2, ::2]
    assert cells.shape == (8, 8)
    assert np.all(cells == SPACE)


def test_maze_contains_walls_between_cells():
    grid = np.asarray(generate_maze(jax.random.PRNGKey(1), 15))

    assert np.any(grid == WALL)


def test_generation_is_deterministic():
    first = np.asarray(generate_maze(jax.random.PRNGKey(42), 15))
    second = np.asarray(generate_maze(jax.random.PRNGKey(42), 15))

    assert np.array_equal(first, second)


def test_generation_differs_across_keys():
    first = np.asarray(generate_maze(jax.random.PRNGKey(1), 15))
    second = np.asarray(generate_maze(jax.random.PRNGKey(2), 15))

    assert not np.array_equal(first, second)


def test_env_rejects_even_world_dim():
    with pytest.raises(ValueError, match='odd'):
        MazeEnv(world_dim=14)


def test_env_rejects_tiny_world_dim():
    with pytest.raises(ValueError, match='odd'):
        MazeEnv(world_dim=1)


def test_reset_structure():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    assert state.env_state.grid.shape == (15, 15)
    assert state.env_state.agent.shape == (2,)
    assert state.env_state.steps.shape == ()
    assert state.done.shape == ()
    assert not bool(state.done)
    assert int(state.env_state.steps) == 0
    assert np.array_equal(np.asarray(state.env_state.agent), [0, 0])


def test_reset_places_exactly_one_goal():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    assert int(jnp.sum(state.env_state.grid == GOAL)) == 1


def test_goal_is_reachable_from_the_agent():
    env = MazeEnv(world_dim=15)
    for seed in range(5):
        state = env.reset(jax.random.PRNGKey(seed))
        grid = np.asarray(state.env_state.grid)

        reachable = bfs_reachable(grid, (0, 0))
        goal = tuple(np.argwhere(grid == GOAL)[0])
        assert goal in reachable, seed


def test_reset_is_deterministic():
    env = MazeEnv(world_dim=15)

    first = env.reset(jax.random.PRNGKey(7))
    second = env.reset(jax.random.PRNGKey(7))

    assert np.array_equal(np.asarray(first.env_state.grid), np.asarray(second.env_state.grid))


def test_reset_differs_across_keys():
    env = MazeEnv(world_dim=15)

    first = env.reset(jax.random.PRNGKey(7))
    second = env.reset(jax.random.PRNGKey(8))

    assert not np.array_equal(np.asarray(first.env_state.grid), np.asarray(second.env_state.grid))
