"""ProcGen's ``climber``: a vertical platformer with crystals and swimming enemies.

Rules from ``procgen/src/games/climber.cpp``: platforms climb a 20x64 shaft, each may
hold a crystal, enemies patrol four tiles either side of where they spawned, and
collecting every crystal ends the level for ten.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.climber import (
    AGENT_RADIUS,
    COIN,
    COIN_REWARD,
    COMPLETION_BONUS,
    ENEMY,
    ENEMY_RADIUS,
    MAX_ENTITIES,
    MAX_JUMP,
    SPACE,
    TIMEOUT,
    WALL_MID,
    WALL_TOP,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    ClimberEnv,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
JIT_ENV = ClimberEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, grid=None, quota=0, collected=0, patrol=None):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
        coin_quota=jnp.asarray(quota, jnp.int32),
        coins_collected=jnp.asarray(collected, jnp.int32),
        patrol_origin=base.patrol_origin if patrol is None else patrol,
    )
    if grid is not None:
        board = board._replace(grid=grid)
    return env.state_from(board)


def put_agent(x=10.0, y=1.5):
    return engine.entity(
        x,
        y,
        rx=AGENT_RADIUS,
        ry=AGENT_RADIUS,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        render_z=1,
        smart_step=True,
    )


def open_grid():
    return jnp.full((WORLD_HEIGHT * WORLD_WIDTH,), SPACE, jnp.int32)


def floor_row(row=0, flat=None):
    flat = open_grid() if flat is None else flat
    return flat.at[row * WORLD_WIDTH : (row + 1) * WORLD_WIDTH].set(WALL_TOP)


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


def test_reset_builds_a_walled_shaft(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid).reshape(WORLD_HEIGHT, WORLD_WIDTH)

    assert set(np.unique(grid).tolist()) <= {WALL_MID, WALL_TOP, SPACE}
    assert np.all(grid[-1] == WALL_MID), 'ceiling'
    assert np.all(grid[:, 0] == WALL_MID) and np.all(grid[:, -1] == WALL_MID), 'sides'
    assert WALL_TOP in grid[0].tolist(), 'the floor is walkable wall tops'


def test_reset_scatters_crystals(env):
    for seed in range(5):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        types = np.asarray(pool.type)[np.asarray(pool.active)].tolist()

        assert types.count(COIN) >= 1, 'the last platform always gets one'
        assert int(state.env_state.coin_quota) == types.count(COIN)


def test_level_seed_pins_the_shaft(env):
    def pinned(seed):
        return ClimberEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))


# --------------------------------------------------------------------------- #
# movement
# --------------------------------------------------------------------------- #
def test_jumping_lifts_the_player(env):
    state = state_with(env, agent=put_agent(10.0, 1.5), grid=floor_row())

    board, _, _ = step_raw(env, state, DOWN)

    assert float(board.agent.vy) == pytest.approx(MAX_JUMP)
    assert bool(board.has_support)


def test_gravity_pulls_the_player_down(env):
    state = state_with(env, agent=put_agent(10.0, 20.5), grid=open_grid())
    board = state.env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    assert float(board.agent.y) < 20.5


def test_the_player_walks_right(env):
    state = state_with(env, agent=put_agent(10.0, 1.5), grid=floor_row())
    board = state.env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))

    assert float(board.agent.x) > 10.0


def test_the_shaft_walls_hold_the_player(env):
    grid = floor_row().reshape(WORLD_HEIGHT, WORLD_WIDTH).at[:, -1].set(WALL_MID).reshape(-1)
    state = state_with(env, agent=put_agent(10.0, 2.5), grid=grid)
    board = state.env_state
    for _ in range(20):
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))

    assert float(board.agent.x) < WORLD_WIDTH - 1.0, 'the right wall stops it'


# --------------------------------------------------------------------------- #
# crystals and swimmers
# --------------------------------------------------------------------------- #
def test_a_crystal_pays_one(env):
    coin = engine.entity(10.0, 1.5, rx=0.3, ry=0.3, type=COIN, image_type=COIN, active=True)
    state = state_with(env, [coin], put_agent(10.0, 1.5), floor_row(), quota=3, collected=1)

    board, reward, _ = step_raw(env, state, NOOP)

    assert reward == pytest.approx(COIN_REWARD)
    assert int(board.coins_collected) == 2
    assert not bool(board.entities.active[0])


def test_the_last_crystal_wins(env):
    coin = engine.entity(10.0, 1.5, rx=0.3, ry=0.3, type=COIN, image_type=COIN, active=True)
    state = state_with(env, [coin], put_agent(10.0, 1.5), floor_row(), quota=1, collected=0)

    _, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert reward == pytest.approx(COIN_REWARD + COMPLETION_BONUS)


def test_an_enemy_ends_the_run(env):
    enemy = engine.entity(
        10.0,
        1.5,
        rx=ENEMY_RADIUS,
        ry=ENEMY_RADIUS,
        type=ENEMY,
        image_type=6,
        smart_step=True,
        active=True,
    )
    state = state_with(env, [enemy], put_agent(10.0, 1.5), floor_row(), quota=3)

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


def test_swimmers_turn_around_at_the_patrol_edge(env):
    enemy = engine.entity(
        20.0,
        5.5,
        1.0,
        0.0,
        rx=ENEMY_RADIUS,
        ry=ENEMY_RADIUS,
        type=ENEMY,
        image_type=6,
        smart_step=True,
        active=True,
    )
    patrol = jnp.zeros((MAX_ENTITIES,), jnp.float32).at[0].set(10.0)
    state = state_with(env, [enemy], put_agent(2.0, 1.5), open_grid(), quota=3, patrol=patrol)

    board, _, _ = step_raw(env, state, NOOP)

    assert float(board.entities.vx[0]) < 0, 'past spawn + PATROL_RANGE it turns back'


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = ClimberEnv(timeout=5)
    state = state_with(env, agent=put_agent(), grid=open_grid(), quota=3)
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_observation_shows_the_shaft(env):
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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([RIGHT, DOWN, NOOP]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
