"""ProcGen's ``starpilot``: a side-scrolling shooter on a spawn timeline.

Rules from ``procgen/src/games/starpilot.cpp``: enemies arrive on a clock that runs
to step 500, every kill pays one point, and the rocket that appears at step 500 ends
the level for ten more.  Touching anything lethal ends the episode on the spot.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.starpilot import (
    BULLET2,
    BULLET_PLAYER,
    CLOUD,
    COMPLETION_BONUS,
    ENEMY_REWARD,
    FINISH_LINE,
    FLYER,
    MAX_ENTITIES,
    MAX_SPAWNERS,
    METEOR,
    SHOOTER_WIN_TIME,
    TIMEOUT,
    TURRET,
    WORLD_DIM,
    StarPilotEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
FIRE_RIGHT, FIRE_LEFT = 9, 10
JIT_ENV = StarPilotEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def ship(x, y, kind, health=1.0, fire_time=0, spawn_time=0, radius=0.5):
    return engine.entity(
        x,
        y,
        rx=radius,
        ry=radius,
        type=kind,
        image_type=kind,
        health=health,
        fire_time=fire_time,
        spawn_time=spawn_time,
        active=True,
    )


def bullet(x, y, vx=0.8, vy=0.0):
    return engine.entity(
        x,
        y,
        vx,
        vy,
        rx=0.4,
        ry=0.4,
        type=BULLET_PLAYER,
        image_type=BULLET_PLAYER,
        collides_with_entities=True,
        active=True,
    )


@pytest.fixture(scope='module')
def env():
    return JIT_ENV


# --------------------------------------------------------------------------- #
# interface and the spawn timeline
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)


def test_reset_builds_an_empty_arena(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert np.all(np.asarray(state.env_state.grid) == SPACE)
    assert not bool(state.env_state.entities.active.any()), 'nothing is on screen yet'
    assert np.allclose([float(state.env_state.agent.x), float(state.env_state.agent.y)], 8.0)


def test_the_timeline_is_sorted_and_runs_to_the_win_time(env):
    state = jit_reset(jax.random.PRNGKey(0))
    spawners = state.env_state.spawners
    active = np.asarray(spawners.active)
    times = np.asarray(spawners.spawn_time)[active]

    assert active.sum() > 10
    assert np.all(times >= 1)
    assert np.all(times <= SHOOTER_WIN_TIME + 4 * 5), 'a group can spawn a few steps late'
    assert np.all(np.diff(times) <= 0), 'the C++ sorts spawners latest first'


def test_the_timeline_only_holds_real_enemy_types(env):
    for seed in range(4):
        state = jit_reset(jax.random.PRNGKey(seed))
        spawners = state.env_state.spawners
        types = set(np.asarray(spawners.type)[np.asarray(spawners.active)].tolist())

        assert types <= {FLYER, METEOR, CLOUD, TURRET, 8}, 'bullets never enter the timeline'
        assert types, 'a timeline with no enemies would be empty'


def test_the_timeline_fills_a_bounded_table(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert state.env_state.spawners.spawn_time.shape == (MAX_SPAWNERS,)


def test_level_seed_pins_the_timeline(env):
    def pinned(seed):
        return StarPilotEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state.spawners

    first, second = pinned(7), pinned(7)

    assert np.array_equal(np.asarray(first.spawn_time), np.asarray(second.spawn_time))
    assert np.array_equal(np.asarray(first.type), np.asarray(second.type))


# --------------------------------------------------------------------------- #
# spawning
# --------------------------------------------------------------------------- #
def test_enemies_appear_when_their_time_comes(env):
    state = jit_reset(jax.random.PRNGKey(0))
    spawners = state.env_state.spawners
    first_time = int(np.max(np.asarray(spawners.spawn_time)[np.asarray(spawners.active)]))

    board = state.env_state
    for _ in range(int(first_time)):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    due = int(np.asarray(spawners.active & (spawners.spawn_time <= first_time)).sum())
    assert int(np.asarray(board.entities.active).sum()) >= due


# ``step_env_state`` without the auto-reset: the tests drive the rules directly, so a
# run should never be replaced underneath them. Jitted because the eager version walks
# 160 entity slots per call.
jit_raw = jax.jit(JIT_ENV.step_env_state)


def step_raw(env, state, action=NOOP):
    board, reward, terminated = jit_raw(state.env_state, jnp.asarray(action))
    return board, float(reward), bool(terminated)


def test_no_enemy_arrives_early(env):
    state = jit_reset(jax.random.PRNGKey(0))

    for _ in range(3):
        state = jit_step(state, jnp.asarray(NOOP))

    earliest = int(np.min(np.asarray(state.env_state.spawners.spawn_time)[np.asarray(state.env_state.spawners.active)]))
    if earliest > 3:
        assert not bool(state.env_state.entities.active.any())


# --------------------------------------------------------------------------- #
# the player's gun
# --------------------------------------------------------------------------- #
def test_firing_spawns_a_bullet(env):
    state = state_with(
        env,
        agent=engine.entity(8.0, 8.0, rx=0.5, ry=0.5, type=engine.PLAYER, image_type=engine.PLAYER, smart_step=True),
    )

    board, _, _ = step_raw(env, state, FIRE_RIGHT)
    pool = board.entities
    shots = pool.active & (pool.type == BULLET_PLAYER)

    assert bool(shots.any()), 'firing spawns a bullet at the ship'


def test_the_bullet_travels_the_way_the_action_points(env):
    agent = engine.entity(8.0, 8.0, rx=0.5, ry=0.5, type=engine.PLAYER, image_type=engine.PLAYER, smart_step=True)
    for action, sign in ((FIRE_RIGHT, 1), (FIRE_LEFT, -1)):
        state = state_with(env, agent=agent)
        stepped = jit_step(state, jnp.asarray(action))
        pool = stepped.env_state.entities
        index = int(np.argmax(np.asarray(pool.active) & (np.asarray(pool.type) == BULLET_PLAYER)))
        if bool(np.asarray(pool.active)[index]):
            assert np.sign(float(pool.vx[index])) == sign


def test_a_bullet_damages_a_flyer(env):
    state = state_with(env, entities=[ship(2.0, 2.0, FLYER, health=2.0), bullet(2.0, 2.0)])

    board, _, _ = step_raw(env, state)

    assert float(board.entities.health[0]) == pytest.approx(1.0), 'a flyer needs two hits'


def test_killing_an_enemy_pays_one_point(env):
    state = state_with(env, entities=[ship(2.0, 2.0, FLYER, health=1.0), bullet(2.0, 2.0)])

    board, reward, _ = step_raw(env, state)

    assert reward == pytest.approx(ENEMY_REWARD)
    assert not bool(board.entities.active[0]), 'the flyer is gone'


def test_meteors_take_five_hundred_hits(env):
    state = state_with(env, entities=[ship(2.0, 2.0, METEOR, health=500.0, radius=2.0), bullet(2.0, 2.0)])

    board, reward, _ = step_raw(env, state)

    assert float(board.entities.health[0]) == pytest.approx(499.0)
    assert reward == 0.0


def test_clouds_are_harmless_props(env):
    state = state_with(env, entities=[ship(2.0, 2.0, CLOUD, health=0.0, radius=2.0), bullet(2.0, 2.0)])

    board, reward, terminated = step_raw(env, state)

    assert bool(board.entities.active[0]), 'clouds are not destructible'
    assert reward == 0.0
    assert not terminated


# --------------------------------------------------------------------------- #
# enemies shoot back
# --------------------------------------------------------------------------- #
def test_a_flyer_fires_once_at_its_fire_time(env):
    state = state_with(env, entities=[ship(2.0, 8.0, FLYER, health=2.0, fire_time=3, spawn_time=0)], steps=2)

    board, _, _ = step_raw(env, state)
    pool = board.entities
    enemy_bullets = pool.active & (pool.type == BULLET2)

    assert bool(enemy_bullets.any()), 'step 3 is spawn_time + fire_time'


def test_a_turret_fires_periodically(env):
    state = state_with(
        env, entities=[ship(2.0, 8.0, TURRET, health=5.0, fire_time=2, spawn_time=0, radius=2.0)], steps=3
    )

    board, _, _ = step_raw(env, state)
    pool = board.entities

    assert bool((pool.active & (pool.type == 3)).any()), 'turrets use BULLET3'


def test_touching_an_enemy_ends_the_episode(env):
    state = state_with(env, entities=[ship(8.0, 8.0, FLYER, health=2.0)])

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])


# --------------------------------------------------------------------------- #
# the finish line
# --------------------------------------------------------------------------- #
def test_the_rocket_appears_at_the_win_time(env):
    state = state_with(env, steps=SHOOTER_WIN_TIME - 1)

    board, _, _ = step_raw(env, state)
    pool = board.entities

    assert bool((pool.active & (pool.type == FINISH_LINE)).any())


def test_reaching_the_rocket_pays_the_bonus(env):
    state = state_with(env, entities=[ship(8.0, 8.0, FINISH_LINE, radius=2.0)])

    board, reward, terminated = step_raw(env, state)

    assert reward == pytest.approx(COMPLETION_BONUS)
    assert terminated


def test_the_win_time_matches_the_original():
    assert SHOOTER_WIN_TIME == 500
    assert TIMEOUT == 1000


def test_timeout_truncates():
    env = StarPilotEnv(timeout=5)
    state = state_with(env, steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


# --------------------------------------------------------------------------- #
# observations, jit, vmap
# --------------------------------------------------------------------------- #
def test_observation_shows_stars_and_ships(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert len({tuple(pixel) for pixel in obs.reshape(-1, 3)}) > 5


def test_a_rollout_runs(env):
    state = jit_reset(jax.random.PRNGKey(0))
    actions = jnp.full((64,), NOOP)

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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, FIRE_RIGHT, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)


def test_the_world_is_sixteen_tiles(env):
    assert WORLD_DIM == 16
