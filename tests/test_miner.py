"""ProcGen's ``miner``: Boulder Dash in a 20x20 block of dirt.

Rules from ``procgen/src/games/miner.cpp``: the robot digs one tile per action,
diamonds and boulders fall and roll, a falling one crushes the robot, and the exit
only opens once every diamond is out.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.miner import (
    AGENT_RADIUS,
    BOULDER,
    COMPLETION_BONUS,
    DIAMOND,
    DIAMOND_REWARD,
    DIRT,
    EXIT,
    MAX_ENTITIES,
    MOVING_BOULDER,
    NUM_BOULDERS,
    NUM_DIAMONDS,
    SPACE,
    TIMEOUT,
    WORLD_DIM,
    MinerEnv,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
JIT_ENV = MinerEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def index_of(x, y):
    return y * WORLD_DIM + x


def state_with(env, grid, agent=None, entities=(), diamonds=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        grid=grid,
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
        diamonds_remaining=jnp.asarray(diamonds, jnp.int32),
    )
    return env.state_from(board)


def put_agent(x, y):
    return engine.entity(
        x + 0.5,
        y + 0.5,
        rx=AGENT_RADIUS,
        ry=AGENT_RADIUS,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        render_z=1,
        smart_step=True,
    )


def dirt_grid():
    return jnp.full((WORLD_DIM * WORLD_DIM,), DIRT, jnp.int32)


def step_raw(env, state, action=NOOP):
    board, reward, terminated = jit_raw(state.env_state, jnp.asarray(action))
    return board, float(reward), bool(terminated)


@pytest.fixture(scope='module')
def env():
    return JIT_ENV


# --------------------------------------------------------------------------- #
# interface and level generation
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)


def test_reset_buries_diamonds_and_boulders(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid)

    assert grid.shape == (WORLD_DIM * WORLD_DIM,)
    assert int((grid == DIAMOND).sum()) == NUM_DIAMONDS
    assert int((grid == BOULDER).sum()) <= NUM_BOULDERS, 'a few are buried next to the robot'
    assert int(state.env_state.diamonds_remaining) == NUM_DIAMONDS


def test_the_robot_starts_in_an_open_cell(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        cell = index_of(int(board.agent.x), int(board.agent.y))

        assert int(board.grid[cell]) == SPACE


def test_the_exit_sits_in_a_cell(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        pool = board.entities
        exit_index = int(np.argmax(np.asarray(pool.type) == EXIT))
        cell = index_of(int(pool.x[exit_index]), int(pool.y[exit_index]))

        assert bool(np.asarray(pool.active)[exit_index])
        assert int(board.grid[cell]) == SPACE


def test_level_seed_pins_the_mine(env):
    def pinned(seed):
        return MinerEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))


# --------------------------------------------------------------------------- #
# digging
# --------------------------------------------------------------------------- #
def test_the_robot_moves_one_tile_per_action(env):
    grid = dirt_grid().at[index_of(10, 5)].set(SPACE).at[index_of(11, 5)].set(SPACE)
    state = state_with(env, grid, put_agent(10, 5), diamonds=3)

    board, _, _ = step_raw(env, state, RIGHT)

    assert float(board.agent.x) == pytest.approx(11.5)


def test_digging_clears_the_tile(env):
    grid = dirt_grid().at[index_of(10, 5)].set(SPACE)
    state = state_with(env, grid, put_agent(10, 5), diamonds=3)

    board, _, _ = step_raw(env, state, NOOP)

    assert int(board.grid[index_of(10, 5)]) == SPACE


def test_a_diamond_pays_one(env):
    grid = dirt_grid().at[index_of(10, 5)].set(DIAMOND)
    state = state_with(env, grid, put_agent(10, 5), diamonds=1)

    board, reward, _ = step_raw(env, state, NOOP)

    assert reward == pytest.approx(DIAMOND_REWARD)
    assert int(board.grid[index_of(10, 5)]) == SPACE
    assert int(board.diamonds_remaining) == 0


def test_a_boulder_blocks_the_robot(env):
    grid = dirt_grid().at[index_of(10, 5)].set(SPACE).at[index_of(11, 5)].set(BOULDER)
    state = state_with(env, grid, put_agent(10, 5), diamonds=3)

    board, _, _ = step_raw(env, state, RIGHT)

    assert float(board.agent.x) == pytest.approx(10.5)


# --------------------------------------------------------------------------- #
# gravity
# --------------------------------------------------------------------------- #
def test_a_boulder_falls_into_empty_space(env):
    grid = dirt_grid().at[index_of(10, 5)].set(BOULDER).at[index_of(10, 4)].set(SPACE)
    state = state_with(env, grid, put_agent(1, 10), diamonds=3)

    board, _, _ = step_raw(env, state, NOOP)

    assert int(board.grid[index_of(10, 4)]) == MOVING_BOULDER
    assert int(board.grid[index_of(10, 5)]) == SPACE


def test_a_boulder_rolls_off_another(env):
    grid = (
        dirt_grid()
        .at[index_of(10, 5)]
        .set(BOULDER)
        .at[index_of(10, 4)]
        .set(BOULDER)
        .at[index_of(9, 4)]
        .set(SPACE)
        .at[index_of(9, 5)]
        .set(SPACE)
    )
    state = state_with(env, grid, put_agent(1, 10), diamonds=3)

    board, _, _ = step_raw(env, state, NOOP)

    assert int(board.grid[index_of(9, 5)]) == BOULDER, 'it rolled to the left'


def test_a_falling_boulder_crushes_the_robot(env):
    grid = dirt_grid().at[index_of(10, 5)].set(MOVING_BOULDER).at[index_of(10, 4)].set(SPACE)
    state = state_with(env, grid, put_agent(10, 4), diamonds=3)

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


# --------------------------------------------------------------------------- #
# the exit
# --------------------------------------------------------------------------- #
def test_the_exit_is_shut_while_diamonds_remain(env):
    cell = index_of(5, 5)
    exit_entity = engine.entity(5.5, 5.5, rx=0.5, ry=0.5, type=EXIT, image_type=EXIT, render_z=-1, active=True)
    # the count is recomputed from the grid, so a diamond has to be buried somewhere
    grid = dirt_grid().at[cell].set(SPACE).at[index_of(15, 15)].set(DIAMOND)
    state = state_with(env, grid, put_agent(5, 5), [exit_entity], diamonds=1)

    _, reward, terminated = step_raw(env, state, NOOP)

    assert not terminated
    assert reward == 0.0


def test_the_exit_pays_once_the_mine_is_empty(env):
    cell = index_of(5, 5)
    exit_entity = engine.entity(5.5, 5.5, rx=0.5, ry=0.5, type=EXIT, image_type=EXIT, render_z=-1, active=True)
    grid = dirt_grid().at[cell].set(SPACE)
    state = state_with(env, grid, put_agent(5, 5), [exit_entity], diamonds=0)

    _, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert reward == pytest.approx(COMPLETION_BONUS)


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = MinerEnv(timeout=5)
    grid = dirt_grid().at[index_of(10, 5)].set(SPACE)
    state = state_with(env, grid, put_agent(10, 5), diamonds=3)
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_observation_shows_the_mine(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert len({tuple(pixel) for pixel in obs.reshape(-1, 3)}) > 5


def test_a_whole_episode_runs_under_scan(env):
    state = jit_reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 15)

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, carry.reward

    final, rewards = jax.jit(lambda: jax.lax.scan(body, state, actions))()

    assert rewards.shape == (64,)
    assert final.obs.shape == (64, 64, 3)


def test_vmap_reset_and_step(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    states = jax.vmap(jax.jit(env.reset))(keys)
    assert states.obs.shape == (3, 64, 64, 3)

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([RIGHT, LEFT, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
