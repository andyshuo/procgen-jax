"""Record one episode of each game.

    pixi run video           # -> runs/*.gif   (ignored by git)
    pixi run video docs      # -> docs/*.gif   (the copies in the README)

Frames are produced by a single jitted ``lax.scan`` and handed to ``ffmpeg`` as
raw RGB, so nothing beyond JAX/NumPy is needed.

``coinrun`` gets two recordings: the agent's own 64x64 observation, and a
two-panel debug view with the whole 64x64 level on the left.  The policy walks
right and jumps over walls and gaps, which is enough to reach the coin.
"""

import sys
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from procgen_jax import CoinRunEnv, MazeEnv, engine
from procgen_jax.games.coinrun import GOAL as COIN
from procgen_jax.games.coinrun import WALL_MID, WALL_TOP, WORLD_DIM, scene_entities
from procgen_jax.objects import GOAL as MAZE_GOAL
from procgen_jax.objects import WALL as MAZE_WALL
from procgen_jax.recording import encode_video, rollout
from procgen_jax.render import render_scene

STEPS = 240
FPS = 20
RIGHT, JUMP_FORWARD = 7, 8


def walk_right_and_jump(state):
    """Jump when the tile ahead is solid or the floor ahead runs out."""
    grid = state.env_state.grid
    agent = state.env_state.agent
    ahead = engine.tile_at(grid, agent.x + 0.7, agent.y, WALL_MID)
    floor_ahead = engine.tile_at(grid, agent.x + 0.9, agent.y - (agent.ry + 0.1), WALL_MID)
    solid = (ahead == WALL_MID) | (ahead == WALL_TOP)
    ledge = ~((floor_ahead == WALL_MID) | (floor_ahead == WALL_TOP))
    return jnp.where(solid | ledge, JUMP_FORWARD, RIGHT)


def longest_level_key(env, candidates=64):
    """Pick the seed whose coin sits furthest right -- the longest level to show."""
    keys = jax.random.split(jax.random.PRNGKey(0), candidates)
    grids = jax.vmap(env.reset)(keys).env_state.grid
    columns = jnp.arange(WORLD_DIM)[None, None, :]
    coin_x = jnp.where(grids == COIN, columns, -1).max(axis=(1, 2))
    return keys[int(jnp.argmax(coin_x))]


def maze_solution_actions(env, steps):
    """Walk the shortest path to the goal, then stand still."""
    state = env.reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid)
    row, col = np.argwhere(grid == MAZE_GOAL)[0]  # argwhere yields (y, x)
    goal = (int(col), int(row))

    frontier, seen = deque([(0, 0)]), {(0, 0)}
    came_from = {}
    while frontier:
        cell = frontier.popleft()
        if cell == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen or not (0 <= nxt[1] < grid.shape[0] and 0 <= nxt[0] < grid.shape[1]):
                continue
            if grid[nxt[1], nxt[0]] == MAZE_WALL:
                continue
            seen.add(nxt)
            came_from[nxt] = (cell, (dx, dy))
            frontier.append(nxt)

    moves = []
    cell = goal
    while cell != (0, 0):
        cell, move = came_from[cell]
        moves.append(move)
    moves.reverse()

    actions = [(dx + 1) * 3 + (dy + 1) for dx, dy in moves]
    return jnp.asarray(actions + [4] * (steps - len(actions)), jnp.int32)


def coinrun_frames(env, policy, key):
    """Agent-view frames and whole-level panels from one scan."""
    state = env.reset(key)

    def body(carry, _):
        carry = env.step(carry, policy(carry))
        return carry, carry

    _, states = jax.lax.scan(body, state, None, length=STEPS)

    agent_view = jnp.repeat(jnp.repeat(states.obs, 4, axis=1), 4, axis=2)
    centre = jnp.asarray([WORLD_DIM / 2, WORLD_DIM / 2])
    whole_level = jax.vmap(
        lambda s: render_scene(s.grid, centre, scene_entities(s), WORLD_DIM, WALL_MID, obs_size=256)
    )(states.env_state)
    return states.obs, jnp.concatenate([whole_level, agent_view], axis=2), states.reward


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')
    out_dir.mkdir(parents=True, exist_ok=True)

    maze = MazeEnv(world_dim=15)
    actions = maze_solution_actions(maze, STEPS)
    frames, rewards, _ = jax.jit(lambda acts: rollout(maze, maze.reset(jax.random.PRNGKey(0)), acts))(actions)
    path = out_dir / 'maze.gif'
    encode_video(path, frames, fps=FPS, scale=4)
    print(f'maze: {STEPS} frames, return {float(rewards.sum()):.1f} -> {path}')

    coinrun = CoinRunEnv()
    key = longest_level_key(coinrun)
    obs, panels, rewards = jax.jit(lambda: coinrun_frames(coinrun, walk_right_and_jump, key))()
    for name, frames, scale in (('coinrun', obs, 4), ('coinrun_map', panels, 1)):
        path = out_dir / f'{name}.gif'
        encode_video(path, frames, fps=FPS, scale=scale)
        print(f'{name}: {STEPS} frames -> {path} ({path.stat().st_size // 1024} KiB)')
    print(f'coinrun return {float(rewards.sum()):.1f}, {int((rewards >= 10).sum())} coin(s) collected')


if __name__ == '__main__':
    main()
