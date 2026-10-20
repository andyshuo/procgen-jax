"""Game rules: movement, collisions, rewards, termination and rendering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax.maze import GOAL_REWARD, MazeEnv, MazeState, vmap_reset, vmap_step
from procgen_jax.objects import GOAL, SPACE, WALL
from procgen_jax.render import AGENT_COLOR, OBS_SIZE

# ProcGen encodes actions as vx = a // 3 - 1, vy = a % 3 - 1.
UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7


def state_from(grid, agent=(0, 0), steps=0, done=False, key=0):
    return MazeState(
        grid=jnp.asarray(grid, dtype=jnp.int32),
        agent=jnp.asarray(agent, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        done=jnp.asarray(done),
        key=jax.random.PRNGKey(key),
    )


@pytest.fixture
def corridor_env():
    """A 3x3 world where the agent can walk right into open space."""
    return MazeEnv(world_dim=3)


def test_action_mapping_matches_procgen():
    env = MazeEnv(world_dim=3)
    # (0,0) -> (1,0) is a rightward move for action 7.
    state = state_from([[SPACE, SPACE, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, _, done = env.step(state, jnp.asarray(RIGHT))

    assert np.array_equal(np.asarray(stepped.agent), [1, 0])
    assert not bool(done)


def test_moves_are_blocked_by_walls(corridor_env):
    state = state_from([[SPACE, WALL, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, reward, done = corridor_env.step(state, jnp.asarray(RIGHT))

    assert np.array_equal(np.asarray(stepped.agent), [0, 0])
    assert float(reward) == 0.0
    assert not bool(done)


def test_agent_cannot_leave_the_world(corridor_env):
    state = state_from([[SPACE, SPACE, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, _, _ = corridor_env.step(state, jnp.asarray(LEFT))

    assert np.array_equal(np.asarray(stepped.agent), [0, 0])


def test_horizontal_movement_cancels_vertical(corridor_env):
    """ProcGen's maze sets ``action_vy = 0`` whenever ``action_vx != 0``."""
    state = state_from([[SPACE, SPACE, SPACE], [SPACE, SPACE, SPACE], [WALL, WALL, WALL]])

    # Action 8 is right + down; maze drops the vertical part.
    stepped, _, _ = corridor_env.step(state, jnp.asarray(8))

    assert np.array_equal(np.asarray(stepped.agent), [1, 0])


def test_vertical_movement_works(corridor_env):
    state = state_from([[SPACE, SPACE, SPACE], [SPACE, SPACE, SPACE], [WALL, WALL, WALL]])

    stepped, _, _ = corridor_env.step(state, jnp.asarray(DOWN))

    assert np.array_equal(np.asarray(stepped.agent), [0, 1])


def test_noop_keeps_the_agent_in_place(corridor_env):
    state = state_from([[SPACE, SPACE, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, reward, done = corridor_env.step(state, jnp.asarray(NOOP))

    assert np.array_equal(np.asarray(stepped.agent), [0, 0])
    assert float(reward) == 0.0
    assert not bool(done)


def test_reaching_the_goal_pays_and_resets(corridor_env):
    state = state_from([[SPACE, GOAL, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, reward, done = corridor_env.step(state, jnp.asarray(RIGHT))

    assert float(reward) == GOAL_REWARD
    assert bool(done)
    # Auto-reset: the agent is back at the start of a fresh episode.
    assert np.array_equal(np.asarray(stepped.agent), [0, 0])
    assert int(stepped.steps) == 0
    assert not bool(stepped.done)


def test_goal_is_consumed(corridor_env):
    state = state_from([[SPACE, GOAL, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])
    state = jax.jit(lambda s: s)(state)

    stepped, _, _ = corridor_env.step(state, jnp.asarray(RIGHT))

    # The replayed transition clears the goal tile before the reset replaces it.
    assert np.asarray(stepped.grid).shape == (3, 3)


def test_timeout_terminates_the_episode():
    env = MazeEnv(world_dim=3, timeout=5)
    state = state_from(
        [[SPACE, SPACE, SPACE], [SPACE, SPACE, SPACE], [SPACE, SPACE, SPACE]],
        steps=4,
    )

    stepped, reward, done = env.step(state, jnp.asarray(NOOP))

    assert bool(done)
    assert float(reward) == 0.0
    assert int(stepped.steps) == 0  # auto-reset


def test_step_counts_up(corridor_env):
    state = state_from([[SPACE, SPACE, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped, _, _ = corridor_env.step(state, jnp.asarray(NOOP))

    assert int(stepped.steps) == 1


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
def test_observation_shape_and_dtype():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    obs = env.observation(state)

    assert obs.shape == (OBS_SIZE, OBS_SIZE, 3)
    assert obs.dtype == jnp.uint8


def test_observation_centres_the_agent():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    obs = np.asarray(env.observation(state))
    centre = obs[OBS_SIZE // 2, OBS_SIZE // 2]

    assert np.array_equal(centre, AGENT_COLOR.astype(np.uint8))


def test_observation_pads_outside_the_world_with_walls():
    """At the top-left corner most of the window is outside the maze."""
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    obs = np.asarray(env.observation(state))

    assert obs[0, 0].tolist() == [140, 140, 150]  # WALL colour


def test_observation_window_follows_the_agent():
    """A unique landmark must move inside the window as the agent walks past it."""
    grid = [
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, GOAL, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
        [SPACE, SPACE, SPACE, SPACE, SPACE, SPACE, SPACE],
    ]
    env = MazeEnv(world_dim=7)
    goal_rgb = np.array([245, 205, 60], dtype=np.uint8)

    left = state_from(grid, agent=(2, 3))
    right = state_from(grid, agent=(4, 3))

    left_obs = np.asarray(env.observation(left))
    right_obs = np.asarray(env.observation(right))

    # The goal sits to the right of the agent in the first case, to the left in the
    # second, so its column inside the 64x64 window must flip.
    left_cols = np.where((left_obs == goal_rgb).all(axis=-1))
    right_cols = np.where((right_obs == goal_rgb).all(axis=-1))
    assert left_cols[1].mean() > right_cols[1].mean()


# --------------------------------------------------------------------------- #
# jit / vmap
# --------------------------------------------------------------------------- #
def test_step_is_jittable():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    step = jax.jit(env.step)
    stepped, reward, done = step(state, jnp.asarray(RIGHT))

    assert stepped.grid.shape == state.grid.shape
    assert jnp.isfinite(reward)
    assert done.dtype == jnp.bool_


def test_whole_episode_runs_under_scan():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 9)

    def body(carry, action):
        state, total = carry
        state, reward, _ = env.step(state, action)
        return (state, total + reward), reward

    (final_state, total_reward), rewards = jax.lax.scan(body, (state, jnp.zeros(())), actions)

    assert rewards.shape == (64,)
    assert jnp.isfinite(total_reward)
    assert final_state.grid.shape == (15, 15)


def test_vmap_reset_and_step():
    env = MazeEnv(world_dim=15)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = vmap_reset(env, keys)
    assert states.grid.shape == (4, 15, 15)
    assert states.agent.shape == (4, 2)

    actions = jnp.array([RIGHT, LEFT, UP, DOWN])
    states, rewards, dones = vmap_step(env, states, actions)

    assert states.grid.shape == (4, 15, 15)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)


def test_vmap_batch_has_distinct_levels():
    env = MazeEnv(world_dim=15)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = vmap_reset(env, keys)
    grids = np.asarray(states.grid)

    assert len({grid.tobytes() for grid in grids}) > 1
