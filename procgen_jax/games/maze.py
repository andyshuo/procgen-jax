"""A pure-JAX port of ProcGen's ``maze`` game.

Rules follow ``procgen/src/games/maze.cpp`` and ``procgen/src/mazegen.cpp``:

* levels are perfect mazes built with Kruskal's algorithm on an odd-sized grid;
* the agent starts in the top-left cell and moves one cell per step;
* actions are encoded as ``vx = a // 3 - 1``, ``vy = a % 3 - 1`` over a 3x3 grid,
  and horizontal movement cancels vertical movement;
* reaching the goal yields +10 and ends the episode; the timeout is 500 steps.

Deliberate first-milestone deviations:

* the maze always fills the world (ProcGen randomises ``maze_dim`` per episode);
* rendering uses flat tiles instead of Kenney sprite assets;
* the action space keeps ProcGen's 15 slots, but only 0-8 are meaningful.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import GOAL, SPACE, WALL
from procgen_jax.render import render_observation

#: ProcGen's ``maze`` rewards 10 for reaching the cheese.
GOAL_REWARD = 10.0
#: ProcGen's ``maze`` has ``timeout = 500``.
TIMEOUT = 500
#: ProcGen exposes 15 discrete actions; 0-8 are the 3x3 movement grid.
NUM_ACTIONS = 15


class MazeState(NamedTuple):
    """Game-specific state: everything needed to continue an episode."""

    grid: jax.Array  # (world_dim, world_dim) int32 of WALL/SPACE/GOAL
    agent: jax.Array  # (2,) int32, (x, y) tile coordinates
    steps: jax.Array  # () int32
    key: jax.Array  # (2,) uint32


# --------------------------------------------------------------------------- #
# level generation
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class MazeEnv(Env):
    """ProcGen's ``maze`` game as a functional, jit-friendly environment."""

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
        # Accepted for API compatibility with ProcGen's env kwargs; levels are
        # derived from the episode key instead of a level id.
        self.start_level = start_level
        self.num_levels = num_levels

    # -- Env interface ----------------------------------------------------- #
    @property
    def action_size(self) -> int:
        return NUM_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (64, 64, 3)

    def reset(self, key: jax.Array) -> State:
        return self.state_from(self.reset_env_state(key))

    def step(self, state: State, action: jax.Array) -> State:
        env_state, reward, terminated = self.step_env_state(state.env_state, action)

        # Brax keeps the two exclusive: a step that terminates is not also
        # truncated, so ``truncated`` unambiguously means "time ran out".
        truncated = jnp.logical_and(env_state.steps >= self.timeout, jnp.logical_not(terminated))
        done = jnp.logical_or(terminated, truncated)

        def reset(env_state: MazeState) -> MazeState:
            reset_key, _ = jax.random.split(env_state.key)
            return self.reset_env_state(reset_key)

        # The terminal frame of the episode is the one the agent acted on. It is
        # free to keep (``state.obs``) and, unlike a post-move render, it still
        # shows the goal tile the agent just stepped onto -- the agent sprite
        # covers the goal once they share a tile.
        return self.state_from(
            auto_reset(env_state, done, reset),
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_obs=state.obs,
        )

    # -- game rules -------------------------------------------------------- #
    def reset_env_state(self, key: jax.Array) -> MazeState:
        maze_key, goal_key = jax.random.split(key)
        grid = generate_maze(maze_key, self.world_dim)

        cells = (self.world_dim + 1) // 2
        goal_cell = jax.random.randint(goal_key, (), 0, cells * cells)
        goal_y = (goal_cell // cells) * 2
        goal_x = (goal_cell % cells) * 2

        return MazeState(
            grid=grid.at[goal_y, goal_x].set(GOAL),
            agent=jnp.zeros((2,), dtype=jnp.int32),
            steps=jnp.zeros((), dtype=jnp.int32),
            key=key,
        )

    def step_env_state(self, env_state: MazeState, action: jax.Array):
        """Advance the game state. Returns ``(state, reward, reached_goal)``."""
        action = jnp.asarray(action, dtype=jnp.int32)
        # Slots 9-14 exist only for ProcGen's shared 15-action space. Decoding
        # them through the same 3x3 arithmetic would yield |v| up to 4, i.e. a
        # multi-tile jump straight through walls, so they are no-ops here.
        valid = (action >= 0) & (action < 9)
        vx = jnp.where(valid, action // 3 - 1, 0)
        vy = jnp.where(valid, action % 3 - 1, 0)
        vy = jnp.where(vx != 0, 0, vy)  # maze: horizontal movement wins

        x, y = env_state.agent[0], env_state.agent[1]
        next_x = jnp.clip(x + vx, 0, self.world_dim - 1)
        next_y = jnp.clip(y + vy, 0, self.world_dim - 1)

        blocked = env_state.grid[next_y, next_x] == WALL
        x = jnp.where(blocked, x, next_x)
        y = jnp.where(blocked, y, next_y)

        reached = env_state.grid[y, x] == GOAL
        reward = jnp.where(reached, GOAL_REWARD, 0.0).astype(jnp.float32)

        stepped = MazeState(
            grid=env_state.grid,
            agent=jnp.stack([x, y]).astype(jnp.int32),
            steps=env_state.steps + 1,
            key=env_state.key,
        )
        return stepped, reward, reached

    # -- State construction ------------------------------------------------ #
    def state_from(
        self,
        env_state: MazeState,
        reward: jax.Array | float = 0.0,
        done: jax.Array | bool = False,
        terminated: jax.Array | bool = False,
        truncated: jax.Array | bool = False,
        final_obs: jax.Array | None = None,
    ) -> State:
        """Wrap a game state into a :class:`~procgen_jax.base.State`."""
        obs = render_observation(env_state.grid, env_state.agent, self.world_dim)
        return State(
            env_state=env_state,
            obs=obs,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            done=jnp.asarray(done),
            info={
                'terminated': jnp.asarray(terminated),
                'truncated': jnp.asarray(truncated),
                'final_obs': obs if final_obs is None else final_obs,
            },
        )
