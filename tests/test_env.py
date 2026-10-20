"""Game rules, transition semantics and rendering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax.base import Wrapper
from procgen_jax.games.maze import GOAL_REWARD, MazeEnv, MazeState
from procgen_jax.objects import GOAL, SPACE, WALL
from procgen_jax.render import AGENT_COLOR, OBS_SIZE

# ProcGen encodes actions as vx = a // 3 - 1, vy = a % 3 - 1.
UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7


def make_state(env, grid, agent=(0, 0), steps=0, key=0):
    """Wrap a hand-written grid into the env's ``State``."""
    env_state = MazeState(
        grid=jnp.asarray(grid, dtype=jnp.int32),
        agent=jnp.asarray(agent, dtype=jnp.int32),
        steps=jnp.asarray(steps, dtype=jnp.int32),
        key=jax.random.PRNGKey(key),
    )
    return env.state_from(env_state)


@pytest.fixture
def env():
    return MazeEnv(world_dim=3)


CORRIDOR = [[SPACE, SPACE, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]]
TWO_ROWS = [[SPACE, SPACE, SPACE], [SPACE, SPACE, SPACE], [WALL, WALL, WALL]]


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes():
    env = MazeEnv(world_dim=15)

    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)
    assert env.observation_size == 64 * 64 * 3
    assert env.unwrapped is env


def test_reset_returns_a_state_with_an_observation():
    state = MazeEnv(world_dim=15).reset(jax.random.PRNGKey(0))

    assert state.obs.shape == (64, 64, 3)
    assert state.obs.dtype == jnp.uint8
    assert not bool(state.done)
    assert float(state.reward) == 0.0
    assert 'terminated' in state.info and 'truncated' in state.info


def test_wrapper_delegates():
    env = MazeEnv(world_dim=15)
    wrapped = Wrapper(env)

    assert wrapped.action_size == env.action_size
    assert wrapped.observation_shape == env.observation_shape
    assert wrapped.unwrapped is env
    state = wrapped.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (64, 64, 3)


# --------------------------------------------------------------------------- #
# movement
# --------------------------------------------------------------------------- #
def test_action_mapping_matches_procgen(env):
    state = make_state(env, CORRIDOR)

    stepped = env.step(state, jnp.asarray(RIGHT))

    assert np.array_equal(np.asarray(stepped.env_state.agent), [1, 0])
    assert not bool(stepped.done)


def test_moves_are_blocked_by_walls(env):
    state = make_state(env, [[SPACE, WALL, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]])

    stepped = env.step(state, jnp.asarray(RIGHT))

    assert np.array_equal(np.asarray(stepped.env_state.agent), [0, 0])
    assert float(stepped.reward) == 0.0


def test_agent_cannot_leave_the_world(env):
    stepped = env.step(make_state(env, CORRIDOR), jnp.asarray(LEFT))

    assert np.array_equal(np.asarray(stepped.env_state.agent), [0, 0])


def test_horizontal_movement_cancels_vertical(env):
    """ProcGen's maze zeroes ``vy`` whenever ``vx != 0``."""
    stepped = env.step(make_state(env, TWO_ROWS), jnp.asarray(8))  # right + down

    assert np.array_equal(np.asarray(stepped.env_state.agent), [1, 0])


def test_vertical_movement_works(env):
    stepped = env.step(make_state(env, TWO_ROWS), jnp.asarray(DOWN))

    assert np.array_equal(np.asarray(stepped.env_state.agent), [0, 1])


def test_noop_keeps_the_agent_in_place(env):
    stepped = env.step(make_state(env, CORRIDOR), jnp.asarray(NOOP))

    assert np.array_equal(np.asarray(stepped.env_state.agent), [0, 0])
    assert float(stepped.reward) == 0.0


# --------------------------------------------------------------------------- #
# termination
# --------------------------------------------------------------------------- #
GOAL_RIGHT = [[SPACE, GOAL, SPACE], [WALL, WALL, WALL], [WALL, WALL, WALL]]


def test_reaching_the_goal_rewards_and_terminates(env):
    stepped = env.step(make_state(env, GOAL_RIGHT), jnp.asarray(RIGHT))

    assert float(stepped.reward) == GOAL_REWARD
    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])
    assert not bool(stepped.info['truncated'])


def test_goal_transition_auto_resets(env):
    stepped = env.step(make_state(env, GOAL_RIGHT), jnp.asarray(RIGHT))

    # The returned state is the start of a fresh episode.
    assert np.array_equal(np.asarray(stepped.env_state.agent), [0, 0])
    assert int(stepped.env_state.steps) == 0
    # gymnax semantics: ``done`` describes the transition, so it stays True even
    # though the returned ``env_state`` has already been reset.
    assert bool(stepped.done)


def test_terminal_observation_is_kept_in_info(env):
    stepped = env.step(make_state(env, GOAL_RIGHT), jnp.asarray(RIGHT))

    final = np.asarray(stepped.info['final_obs'])
    assert final.shape == (OBS_SIZE, OBS_SIZE, 3)
    # The reward tile is visible to the right of the agent in the terminal frame.
    assert np.any((final == np.array([245, 205, 60])).all(axis=-1))


def test_timeout_truncates():
    env = MazeEnv(world_dim=3, timeout=5)
    state = make_state(env, TWO_ROWS, steps=4)

    stepped = env.step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])
    assert not bool(stepped.info['terminated'])
    assert float(stepped.reward) == 0.0
    assert int(stepped.env_state.steps) == 0  # auto-reset


def test_steps_counter_increases(env):
    stepped = env.step(make_state(env, CORRIDOR), jnp.asarray(NOOP))

    assert int(stepped.env_state.steps) == 1


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
def test_observation_centres_the_agent():
    env = MazeEnv(world_dim=15)
    obs = np.asarray(env.reset(jax.random.PRNGKey(0)).obs)

    assert np.array_equal(obs[OBS_SIZE // 2, OBS_SIZE // 2], AGENT_COLOR.astype(np.uint8))


def test_observation_pads_outside_the_world_with_walls():
    env = MazeEnv(world_dim=15)
    obs = np.asarray(env.reset(jax.random.PRNGKey(0)).obs)

    assert obs[0, 0].tolist() == [140, 140, 150]


def test_observation_window_follows_the_agent():
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

    left = np.asarray(make_state(env, grid, agent=(2, 3)).obs)
    right = np.asarray(make_state(env, grid, agent=(4, 3)).obs)

    left_cols = np.where((left == goal_rgb).all(axis=-1))[1]
    right_cols = np.where((right == goal_rgb).all(axis=-1))[1]
    assert left_cols.mean() > right_cols.mean()


# --------------------------------------------------------------------------- #
# jit / vmap / scan
# --------------------------------------------------------------------------- #
def test_step_is_jittable():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))

    stepped = jax.jit(env.step)(state, jnp.asarray(RIGHT))

    assert stepped.obs.shape == state.obs.shape
    assert jnp.isfinite(stepped.reward)
    assert stepped.done.dtype == jnp.bool_


def test_whole_episode_runs_under_scan():
    env = MazeEnv(world_dim=15)
    state = env.reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 9)

    def body(carry, action):
        state, total = carry
        stepped = env.step(state, action)
        return (stepped, total + stepped.reward), stepped.reward

    (final_state, total_reward), rewards = jax.lax.scan(body, (state, jnp.zeros(())), actions)

    assert rewards.shape == (64,)
    assert jnp.isfinite(total_reward)
    assert final_state.env_state.grid.shape == (15, 15)


def test_vmap_reset_and_step():
    env = MazeEnv(world_dim=15)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = jax.vmap(env.reset)(keys)
    assert states.obs.shape == (4, 64, 64, 3)
    assert states.env_state.grid.shape == (4, 15, 15)

    actions = jnp.array([RIGHT, LEFT, UP, DOWN])
    stepped = jax.vmap(env.step)(states, actions)

    assert stepped.obs.shape == (4, 64, 64, 3)
    assert stepped.reward.shape == (4,)
    assert stepped.done.shape == (4,)


def test_vmap_batch_has_distinct_levels():
    env = MazeEnv(world_dim=15)
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))

    grids = np.asarray(states.env_state.grid)
    assert len({grid.tobytes() for grid in grids}) > 1
