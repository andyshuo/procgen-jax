"""A pure-JAX port of ProcGen's ``maze`` game.

The rules mirror ``procgen/src/games/maze.cpp`` and ``procgen/src/mazegen.cpp``:

* levels are perfect mazes built with Kruskal's algorithm on an odd-sized grid;
* the agent starts in the top-left cell and moves one cell per step;
* actions are encoded as ``vx = a // 3 - 1``, ``vy = a % 3 - 1`` over a 3x3 grid,
  and horizontal movement cancels vertical movement;
* reaching the goal yields +10 and ends the episode; the timeout is 500 steps.

Known deviations from the original, all deliberate for a first milestone:

* the maze always fills the world (ProcGen randomises ``maze_dim`` and centres it
  inside ``world_dim``);
* rendering uses flat tiles instead of Kenney sprite assets;
* the action space keeps ProcGen's 15 slots, but only 0-8 are meaningful.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax.objects import GOAL, SPACE, WALL
from procgen_jax.render import render_observation

#: ProcGen's ``maze`` rewards 10 for reaching the cheese.
GOAL_REWARD = 10.0
#: ProcGen's ``maze`` has ``timeout = 500``.
TIMEOUT = 500
#: ProcGen exposes 15 discrete actions; 0-8 are the 3x3 movement grid.
NUM_ACTIONS = 15


class MazeState(NamedTuple):
    """Everything needed to continue an episode (a JAX pytree)."""

    grid: jax.Array  # (world_dim, world_dim) int32 of WALL/SPACE/GOAL
    agent: jax.Array  # (2,) int32, (x, y) tile coordinates
    steps: jax.Array  # () int32
    done: jax.Array  # () bool
    key: jax.Array  # (2,) uint32


def _find(parents, node):
    """Root of ``node`` in a union-find array (no path compression)."""

    def cond(current):
        return parents[current] != current

    def body(current):
        return parents[current]

    return jax.lax.while_loop(cond, body, node)


def _wall_list(cells_per_side: int):
    """Enumerate every wall between adjacent cells as ``(a, b, y, x)``.

    Cells live at even coordinates ``(2i, 2j)``; the wall between two horizontally
    adjacent cells sits at ``(2i, 2j + 1)`` and the vertical one at ``(2i + 1, 2j)``.
    """
    m = cells_per_side

    i = jnp.arange(m)[:, None]
    j = jnp.arange(m - 1)[None, :]
    right_a = (i * m + j).reshape(-1)
    right_b = right_a + 1
    right_y = (2 * i + 0 * j).reshape(-1)
    right_x = (0 * i + 2 * j + 1).reshape(-1)

    i = jnp.arange(m - 1)[:, None]
    j = jnp.arange(m)[None, :]
    down_a = (i * m + j).reshape(-1)
    down_b = down_a + m
    down_y = (2 * i + 1 + 0 * j).reshape(-1)
    down_x = (0 * i + 2 * j).reshape(-1)

    return (
        jnp.concatenate([right_a, down_a]),
        jnp.concatenate([right_b, down_b]),
        jnp.concatenate([right_y, down_y]),
        jnp.concatenate([right_x, down_x]),
    )


def generate_maze(key: jax.Array, maze_dim: int) -> jax.Array:
    """Build a perfect maze of size ``maze_dim`` (must be odd) with Kruskal's algorithm."""
    cells_per_side = (maze_dim + 1) // 2
    a, b, wall_y, wall_x = _wall_list(cells_per_side)

    order = jax.random.permutation(key, a.shape[0])
    a, b, wall_y, wall_x = a[order], b[order], wall_y[order], wall_x[order]

    grid = jnp.full((maze_dim, maze_dim), WALL, dtype=jnp.int32)
    parents = jnp.arange(cells_per_side * cells_per_side)

    def open_wall(carry, index):
        grid, parents = carry
        root_a = _find(parents, a[index])
        root_b = _find(parents, b[index])

        def unite(_):
            carved = grid.at[wall_y[index], wall_x[index]].set(SPACE)
            return carved, parents.at[root_a].set(root_b)

        def skip(_):
            return grid, parents

        return jax.lax.cond(root_a != root_b, unite, skip, None), None

    (grid, _), _ = jax.lax.scan(open_wall, (grid, parents), jnp.arange(a.shape[0]))

    # Carve the cell centres themselves (every cell is now reachable).
    ys = (2 * jnp.arange(cells_per_side))[:, None]
    xs = (2 * jnp.arange(cells_per_side))[None, :]
    return grid.at[ys, xs].set(SPACE)


class MazeEnv:
    """ProcGen's ``maze`` game as a functional, jit-friendly environment."""

    num_actions = NUM_ACTIONS
    obs_shape = (64, 64, 3)

    def __init__(
        self,
        world_dim: int = 15,
        timeout: int = TIMEOUT,
        start_level: int = 0,
        num_levels: int = 0,
    ):
        if world_dim < 3 or world_dim % 2 == 0:
            raise ValueError(f'world_dim must be odd and >= 3, got {world_dim}')
        self.world_dim = world_dim
        self.timeout = timeout
        # Accepted for API compatibility with ProcGen's env kwargs; the JAX
        # implementation derives every level from the episode key instead.
        self.start_level = start_level
        self.num_levels = num_levels

    # ------------------------------------------------------------------ #
    # environment API
    # ------------------------------------------------------------------ #
    def reset(self, key: jax.Array) -> MazeState:
        """Start a new episode from ``key``."""
        maze_key, goal_key = jax.random.split(key)
        grid = generate_maze(maze_key, self.world_dim)

        cells = (self.world_dim + 1) // 2
        goal_cell = jax.random.randint(goal_key, (), 0, cells * cells)
        goal_y = (goal_cell // cells) * 2
        goal_x = (goal_cell % cells) * 2
        grid = grid.at[goal_y, goal_x].set(GOAL)

        return MazeState(
            grid=grid,
            agent=jnp.zeros((2,), dtype=jnp.int32),
            steps=jnp.zeros((), dtype=jnp.int32),
            done=jnp.zeros((), dtype=jnp.bool_),
            key=key,
        )

    def step(self, state: MazeState, action: jax.Array):
        """Advance one step. Returns ``(state, reward, done)``.

        ``done`` refers to the transition; ``state`` is already auto-reset when the
        episode ended, matching ProcGen's vectorised-env behaviour.
        """
        action = jnp.asarray(action, dtype=jnp.int32)
        vx = action // 3 - 1
        vy = action % 3 - 1
        vy = jnp.where(vx != 0, 0, vy)  # maze: horizontal movement wins

        x, y = state.agent[0], state.agent[1]
        next_x = jnp.clip(x + vx, 0, self.world_dim - 1)
        next_y = jnp.clip(y + vy, 0, self.world_dim - 1)

        blocked = state.grid[next_y, next_x] == WALL
        x = jnp.where(blocked, x, next_x)
        y = jnp.where(blocked, y, next_y)

        reached = state.grid[y, x] == GOAL
        reward = jnp.where(reached, GOAL_REWARD, 0.0).astype(jnp.float32)
        grid = jnp.where(reached, state.grid.at[y, x].set(SPACE), state.grid)

        steps = state.steps + 1
        done = jnp.logical_or(reached, steps >= self.timeout)

        stepped = MazeState(
            grid=grid,
            agent=jnp.stack([x, y]).astype(jnp.int32),
            steps=steps,
            done=done,
            key=state.key,
        )
        episode_key, _ = jax.random.split(state.key)
        fresh = self.reset(episode_key)
        state = jax.lax.cond(done, lambda _: fresh, lambda _: stepped, None)
        return state, reward, done

    def observation(self, state: MazeState) -> jax.Array:
        """Agent-centred 64x64x3 uint8 view of the maze."""
        return render_observation(state.grid, state.agent, self.world_dim)


# --------------------------------------------------------------------------- #
# vectorised helpers
# --------------------------------------------------------------------------- #
def vmap_reset(env: MazeEnv, keys: jax.Array) -> MazeState:
    return jax.vmap(env.reset)(keys)


def vmap_step(env: MazeEnv, state: MazeState, actions: jax.Array):
    return jax.vmap(env.step)(state, actions)
