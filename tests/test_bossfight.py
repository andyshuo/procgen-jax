"""ProcGen's ``bossfight``: a shielded boss, four attack patterns and meteors.

Rules from ``procgen/src/games/bossfight.cpp``: the boss drifts around the upper part
of a 20x20 arena, its shield opens and closes every few relocations, lasers bounce off
while it is up and take one health each while it is down, and a round pays one every
``round_health`` hits with ten for the kill.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.bossfight import (
    AGENT_RADIUS,
    BARRIER,
    BARRIER_RADIUS,
    BOSS,
    BOSS_BULLET_EXPIRE,
    BOSS_RADIUS,
    BULLET_RADIUS,
    COMPLETION_BONUS,
    ENEMY_BULLET,
    FIRE_COOLDOWN,
    MAX_BARRIERS,
    MAX_ENTITIES,
    PLAYER_BULLET,
    POSITIVE_REWARD,
    REFLECTED_BULLET,
    SHIELDS,
    TIMEOUT,
    WORLD_DIM,
    BossfightEnv,
    attack_shots,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
FIRE = 9
JIT_ENV = BossfightEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities, agent=None, shields_up=True, steps=0, health=None):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        entities=make_pool(entities),
        agent=base.agent if agent is None else agent,
        shields_are_up=jnp.asarray(shields_up),
        steps=jnp.asarray(steps, jnp.int32),
    )
    if health is not None:
        board = board._replace(health=jnp.asarray(health, jnp.int32))
    return env.state_from(board)


def boss_entity(x=10.0, y=14.0, health=7):
    return engine.entity(
        x,
        y,
        rx=BOSS_RADIUS,
        ry=BOSS_RADIUS,
        type=BOSS,
        image_type=BOSS,
        image_theme=0,
        health=health,
        active=True,
    )


def shield_entity(x=10.0, y=14.0):
    return engine.entity(
        x, y, rx=1.2 * BOSS_RADIUS, ry=1.2 * BOSS_RADIUS, type=SHIELDS, image_type=SHIELDS, active=True
    )


def put_agent(x=10.0, y=2.0):
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


def laser_entity(x, y, vx=0.0, vy=1.0):
    return engine.entity(
        x,
        y,
        vx,
        vy,
        rx=BULLET_RADIUS,
        ry=BULLET_RADIUS,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        collides_with_entities=True,
        expire_time=25,
        active=True,
    )


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


def test_reset_builds_the_arena(env):
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert types.tolist().count(BOSS) == 1
    assert types.tolist().count(SHIELDS) == 1
    assert 1 <= types.tolist().count(BARRIER) <= MAX_BARRIERS
    assert bool(state.env_state.shields_are_up)


def test_the_boss_starts_with_rounds_of_health(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state

        assert int(board.health) == int(board.round_health) * int(board.num_rounds)
        assert 1 <= int(board.round_health) <= 9
        assert 1 <= int(board.num_rounds) <= 5


def test_the_player_starts_inside_the_arena(env):
    for seed in range(4):
        agent = jit_reset(jax.random.PRNGKey(seed)).env_state.agent

        assert 0.0 <= float(agent.y) - float(agent.ry)
        assert float(agent.x) + float(agent.rx) < WORLD_DIM


def test_level_seed_pins_the_arena(env):
    def pinned(seed):
        return BossfightEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert int(first.health) == int(second.health)
    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))


# --------------------------------------------------------------------------- #
# the arena edge and the players
# --------------------------------------------------------------------------- #
def test_the_arena_edge_is_solid(env):
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0))
    board = state.env_state
    for _ in range(30):
        board, _, term = jit_raw(board, jnp.asarray(UP))

    assert bool(term) or float(board.agent.y) < WORLD_DIM
    assert float(board.agent.y) >= 0.0, 'the edge stops the ship'


def test_firing_needs_the_cooldown(env):
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0), steps=0)

    early, _, _ = step_raw(env, state, FIRE)
    assert not (np.asarray(early.entities.type) == PLAYER_BULLET)[np.asarray(early.entities.active)].any()

    ready = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0), steps=FIRE_COOLDOWN)
    fired, _, _ = step_raw(env, ready, FIRE)
    assert (np.asarray(fired.entities.type) == PLAYER_BULLET)[np.asarray(fired.entities.active)].any()


# --------------------------------------------------------------------------- #
# lasers against the shield and the boss
# --------------------------------------------------------------------------- #
def test_a_laser_bounces_off_the_shield(env):
    laser = laser_entity(10.0, 14.0)
    state = state_with(env, [boss_entity(), shield_entity(), laser], put_agent(2.0, 2.0), shields_up=True, health=7)

    board, reward, _ = step_raw(env, state, NOOP)

    assert int(board.entities.type[2]) == REFLECTED_BULLET, 'the laser is reflected'
    assert int(board.health) == 7, 'and the boss is untouched'
    assert reward == 0.0


def test_a_laser_hurts_the_boss_behind_an_open_shield(env):
    laser = laser_entity(10.0, 14.0)
    state = state_with(env, [boss_entity(), shield_entity(), laser], put_agent(2.0, 2.0), shields_up=False, health=7)

    board, _, _ = step_raw(env, state, NOOP)

    assert int(board.health) == 6
    assert not bool(board.entities.active[2]), 'the laser is spent'


def test_a_round_pays_one(env):
    laser = laser_entity(10.0, 14.0)
    state = state_with(env, [boss_entity(), shield_entity(), laser], put_agent(2.0, 2.0), shields_up=False, health=8)
    state = state.env_state

    board, reward, _ = jit_raw(state, jnp.asarray(NOOP))

    assert int(board.health) == 7
    assert reward == pytest.approx(POSITIVE_REWARD), 'seven hit them all at once'


def test_the_last_hit_wins(env):
    laser = laser_entity(10.0, 14.0)
    state = state_with(
        env, [boss_entity(health=1), shield_entity(), laser], put_agent(2.0, 2.0), shields_up=False, health=1
    )

    board, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert int(board.health) == 0
    assert reward == pytest.approx(POSITIVE_REWARD + COMPLETION_BONUS)


# --------------------------------------------------------------------------- #
# the boss's guns and the meteors
# --------------------------------------------------------------------------- #
def test_the_boss_fires_in_patterns(env):
    angles, active = jax.jit(attack_shots)(jnp.asarray(0), jnp.asarray(8), jnp.asarray(0.5))
    assert int(np.asarray(active).sum()) == 5, 'the fan is five bullets'

    angles, active = jax.jit(attack_shots)(jnp.asarray(2), jnp.asarray(10), jnp.asarray(0.5))
    assert int(np.asarray(active).sum()) == 8, 'the ring is eight'

    angles, active = jax.jit(attack_shots)(jnp.asarray(1), jnp.asarray(5), jnp.asarray(0.0))
    assert int(np.asarray(active).sum()) == 4, 'the cross is four'


def test_the_boss_shoots_enemy_bullets(env):
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0), steps=7)
    state = state.env_state._replace(attack_mode=jnp.asarray(0, jnp.int32))

    board, _, _ = jit_raw(state, jnp.asarray(NOOP))
    types = np.asarray(board.entities.type)
    active = np.asarray(board.entities.active)

    assert int(((types == ENEMY_BULLET) & active).sum()) == 5


def test_an_enemy_bullet_ends_the_run(env):
    bullet = engine.entity(
        2.0,
        2.0,
        0.0,
        0.0,
        rx=BULLET_RADIUS,
        ry=BULLET_RADIUS,
        type=ENEMY_BULLET,
        image_type=ENEMY_BULLET,
        expire_time=BOSS_BULLET_EXPIRE,
        active=True,
    )
    state = state_with(env, [boss_entity(), shield_entity(), bullet], put_agent(2.0, 2.0))

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


def test_a_meteor_ends_the_run(env):
    rock = engine.entity(
        2.0,
        2.0,
        rx=BARRIER_RADIUS,
        ry=BARRIER_RADIUS,
        type=BARRIER,
        image_type=BARRIER,
        health=3.0,
        collides_with_entities=True,
        active=True,
    )
    state = state_with(env, [boss_entity(), shield_entity(), rock], put_agent(2.0, 2.0))

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


def test_the_shield_follows_the_boss(env):
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0))

    board, _, _ = step_raw(env, state, NOOP)

    assert float(board.entities.x[1]) == pytest.approx(float(board.entities.x[0]))


def test_the_boss_stays_in_its_half(env):
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0))
    board = state.env_state
    for _ in range(80):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    assert float(board.entities.y[0]) >= BOSS_RADIUS - 0.01


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = BossfightEnv(timeout=5)
    state = state_with(env, [boss_entity(), shield_entity()], put_agent(2.0, 2.0), steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 4000


def test_observation_shows_the_arena(env):
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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([RIGHT, FIRE, LEFT]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
