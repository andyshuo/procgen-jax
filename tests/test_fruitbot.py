"""ProcGen's ``fruitbot``: descend a shaft, eat fruit, dodge barriers.

Rules from ``procgen/src/games/fruitbot.cpp``: the vertical action is pinned so the
robot always sinks, fruit pays one, junk costs four, barriers and locked doors end
the run, a present at the bottom pays ten, and action 9 shoots a key that opens a
lock and the door beside it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.fruitbot import (
    AGENT_RADIUS,
    BAD_OBJ,
    BARRIER,
    COMPLETION_BONUS,
    GOOD_OBJ,
    KEY_DURATION,
    LOCK,
    LOCKED_DOOR,
    MAX_ENTITIES,
    NUM_WALLS,
    PENALTY,
    PLAYER_BULLET,
    POSITIVE_REWARD,
    PRESENT,
    TIMEOUT,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    FruitBotEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
FIRE = 9
JIT_ENV = FruitBotEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, last_fire_time=0, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            last_fire_time=jnp.asarray(last_fire_time, jnp.int32),
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def shaft_state(env, entities, agent, **kwargs):
    state = state_with(env, entities=entities, agent=agent, **kwargs)
    return env.state_from(state.env_state._replace(grid=jnp.full((WORLD_HEIGHT, WORLD_WIDTH), SPACE, jnp.int32)))


def put_agent(x=10.0, y=10.0):
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


def item_at(kind, x, y, radius=0.5):
    return engine.entity(x, y, rx=radius, ry=radius, type=kind, image_type=kind, active=True)


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


def test_reset_builds_ten_rows_of_barriers(env):
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert types.tolist().count(BARRIER) == 2 * NUM_WALLS, 'two pieces per row'
    assert np.all(np.asarray(state.env_state.grid) == SPACE)


def test_reset_lays_presents_across_the_bottom(env):
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    presents = active & (np.asarray(pool.type) == PRESENT)

    assert int(presents.sum()) == WORLD_WIDTH
    assert np.allclose(np.asarray(pool.y)[presents], WORLD_HEIGHT - 0.5)


def test_reset_scatters_fruit_and_junk(env):
    for seed in range(4):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        types = np.asarray(pool.type)[np.asarray(pool.active)].tolist()

        assert 10 <= types.count(GOOD_OBJ) <= 19
        assert 10 <= types.count(BAD_OBJ) <= 19


def test_the_robot_starts_at_the_top(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert float(state.env_state.agent.y) == pytest.approx(AGENT_RADIUS)
    assert 0 < float(state.env_state.agent.x) < WORLD_WIDTH


def test_level_seed_pins_the_level(env):
    def pinned(seed):
        return FruitBotEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(5), pinned(5)

    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))
    assert np.array_equal(np.asarray(first.entities.type), np.asarray(second.entities.type))


# --------------------------------------------------------------------------- #
# movement
# --------------------------------------------------------------------------- #
def test_the_robot_sinks_on_its_own(env):
    state = shaft_state(env, [], put_agent(10.0, 5.0))

    for _ in range(20):
        state = jit_step(state, jnp.asarray(NOOP))

    assert float(state.env_state.agent.y) > 5.0, 'the vertical action is pinned'


def test_the_robot_steers_sideways(env):
    state = shaft_state(env, [], put_agent(10.0, 5.0))

    for _ in range(10):
        state = jit_step(state, jnp.asarray(RIGHT))
    right = float(state.env_state.agent.x)
    for _ in range(10):
        state = jit_step(state, jnp.asarray(LEFT))

    assert right > 10.0
    assert float(state.env_state.agent.x) < right


# --------------------------------------------------------------------------- #
# food
# --------------------------------------------------------------------------- #
def test_fruit_pays_one_and_disappears(env):
    state = shaft_state(env, [item_at(GOOD_OBJ, 10.0, 10.0)], put_agent(10.0, 10.0))

    board, reward, _ = step_raw(env, state)

    assert reward == pytest.approx(POSITIVE_REWARD)
    assert not bool(board.entities.active[0])


def test_junk_costs_four_and_disappears(env):
    state = shaft_state(env, [item_at(BAD_OBJ, 10.0, 10.0)], put_agent(10.0, 10.0))

    board, reward, _ = step_raw(env, state)

    assert reward == pytest.approx(PENALTY)
    assert not bool(board.entities.active[0])


def test_a_present_wins_the_level(env):
    state = shaft_state(env, [item_at(PRESENT, 10.0, 10.0)], put_agent(10.0, 10.0))

    _, reward, terminated = step_raw(env, state)

    assert terminated
    assert reward == pytest.approx(COMPLETION_BONUS)


def test_a_barrier_ends_the_episode(env):
    state = shaft_state(
        env,
        [engine.entity(10.0, 10.0, rx=4.0, ry=0.3, type=BARRIER, image_type=BARRIER, active=True)],
        put_agent(10.0, 10.0),
    )

    _, _, terminated = step_raw(env, state)

    assert terminated


def test_a_locked_door_ends_the_episode(env):
    state = shaft_state(
        env,
        [engine.entity(10.0, 10.0, rx=3.0, ry=0.3, type=LOCKED_DOOR, image_type=LOCKED_DOOR, active=True)],
        put_agent(10.0, 10.0),
    )

    _, _, terminated = step_raw(env, state)

    assert terminated


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #
def test_firing_spawns_a_key_that_falls(env):
    state = shaft_state(env, [], put_agent(10.0, 10.0), last_fire_time=0, steps=8)

    board, _, _ = step_raw(env, state, FIRE)
    pool = board.entities
    keys = np.asarray(pool.active) & (np.asarray(pool.type) == PLAYER_BULLET)

    assert bool(keys.any())
    index = int(np.argmax(keys))
    assert float(pool.vy[index]) > 0, 'the key drifts down'


def test_the_key_has_a_cooldown(env):
    state = shaft_state(env, [], put_agent(10.0, 10.0), last_fire_time=4, steps=8)

    board, _, _ = step_raw(env, state, FIRE)

    assert not bool((board.entities.active & (board.entities.type == PLAYER_BULLET)).any())


def test_the_key_expires(env):
    key = engine.entity(
        10.0,
        10.0,
        0.0,
        0.5,
        rx=0.25,
        ry=0.25,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        collides_with_entities=True,
        expire_time=KEY_DURATION,
        active=True,
    )

    for _ in range(KEY_DURATION):
        key = engine.entity_step(key)
    assert bool(key.active)

    for _ in range(2):
        key = engine.entity_step(key)
    assert not bool(key.active)


def test_shooting_a_lock_opens_its_door(env):
    lock = engine.entity(10.0, 10.2, rx=0.25, ry=0.45, type=LOCK, image_type=LOCK, active=True)
    door = engine.entity(10.0, 10.0, rx=3.0, ry=0.3, type=LOCKED_DOOR, image_type=LOCKED_DOOR, active=True)
    key = engine.entity(
        10.0,
        10.2,
        0.0,
        0.0,
        rx=0.25,
        ry=0.25,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        collides_with_entities=True,
        expire_time=KEY_DURATION,
        active=True,
    )
    state = shaft_state(env, [lock, door, key], put_agent(2.0, 2.0))

    board, _, _ = step_raw(env, state, NOOP)
    active = np.asarray(board.entities.active)

    assert not active[0], 'the lock is broken'
    assert not active[1], 'and the door at the same height opens'
    assert not active[2], 'the key is spent'


def test_a_key_stops_on_a_barrier(env):
    wall = engine.entity(10.0, 10.0, rx=4.0, ry=0.3, type=BARRIER, image_type=BARRIER, active=True)
    key = engine.entity(
        10.0,
        10.0,
        0.0,
        0.0,
        rx=0.25,
        ry=0.25,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        collides_with_entities=True,
        expire_time=KEY_DURATION,
        active=True,
    )
    state = shaft_state(env, [wall, key], put_agent(2.0, 2.0))

    board, _, _ = step_raw(env, state, NOOP)

    assert not bool(board.entities.active[1])
    assert bool(board.entities.active[0]), 'the barrier is indestructible'


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = FruitBotEnv(timeout=5)
    state = shaft_state(env, [], put_agent(), steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_the_camera_follows_the_robot(env):
    # A barrier is only in view when the robot is near it, so the frame changes.
    wall = engine.entity(10.0, 40.0, rx=4.0, ry=0.3, type=BARRIER, image_type=BARRIER, active=True)
    low = shaft_state(env, [wall], put_agent(10.0, 40.0))
    high = shaft_state(env, [wall], put_agent(10.0, 5.0))

    low_obs = np.asarray(env.observe(low.env_state))
    high_obs = np.asarray(env.observe(high.env_state))

    assert low_obs.shape == (64, 64, 3)
    assert high_obs.shape == (64, 64, 3)
    assert not np.array_equal(low_obs, high_obs), 'the view tracks the descent'


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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, FIRE, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
