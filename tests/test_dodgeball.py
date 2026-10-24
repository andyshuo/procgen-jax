"""ProcGen's ``dodgeball``: rooms cut by lava, enemies that throw back, and a door.

Rules from ``procgen/src/games/dodgeball.cpp``: the arena is split four times by
lava walls, three to six enemies wander it, action 9 throws a ball on a seven-step
cooldown, each kill pays two, and the exit only opens once every enemy is down.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.dodgeball import (
    AGENT_RADIUS,
    BALL_RADIUS,
    COMPLETION_BONUS,
    DOOR,
    DOOR_OPEN,
    ENEMY,
    ENEMY_BALL,
    ENEMY_REWARD,
    LAVA_WALL,
    MAX_ENTITIES,
    NUM_ITERATIONS,
    PLAYER_BALL,
    TIMEOUT,
    WORLD_DIM,
    DodgeballEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
THROW = 9
JIT_ENV = DodgeballEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, num_enemies=0, last_fire_time=0, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            num_enemies=jnp.asarray(num_enemies, jnp.int32),
            last_fire_time=jnp.asarray(last_fire_time, jnp.int32),
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def open_grid():
    return jnp.zeros((WORLD_DIM, WORLD_DIM), jnp.int32)


def floor_state(env, entities, agent, **kwargs):
    state = state_with(env, entities=entities, agent=agent, **kwargs)
    return env.state_from(state.env_state._replace(grid=open_grid()))


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


def enemy_at(x, y, theme=0):
    return engine.entity(
        x, y, rx=0.75, ry=0.75, type=ENEMY, image_type=ENEMY, image_theme=theme, health=1.0, active=True
    )


def ball_at(x, y, kind=PLAYER_BALL):
    return engine.entity(
        x,
        y,
        rx=BALL_RADIUS,
        ry=BALL_RADIUS,
        type=kind,
        image_type=kind,
        collides_with_entities=True,
        expire_time=50,
        active=True,
    )


def wall_at(x, y, rx=3.0, ry=0.45):
    return engine.entity(
        x, y, rx=rx, ry=ry, type=LAVA_WALL, image_type=LAVA_WALL, collides_with_entities=True, active=True
    )


def door_at(x, y):
    return engine.entity(x, y, rx=0.75, ry=0.75, type=DOOR, image_type=DOOR, active=True)


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


def test_reset_cuts_the_arena_into_rooms(env):
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert np.all(np.asarray(state.env_state.grid) == SPACE)
    assert types.tolist().count(LAVA_WALL) == NUM_ITERATIONS, 'one wall per split'
    assert types.tolist().count(DOOR) == 1


def test_reset_scatters_three_to_six_enemies(env):
    for seed in range(6):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        types = np.asarray(pool.type)[np.asarray(pool.active)]

        assert 3 <= types.tolist().count(ENEMY) <= 6
        assert int(state.env_state.num_enemies) == types.tolist().count(ENEMY)


def test_everything_starts_inside_the_arena(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        pool = board.entities
        active = np.asarray(pool.active)

        assert 0 <= float(board.agent.x) <= WORLD_DIM
        assert 0 <= float(board.agent.y) <= WORLD_DIM
        xs = np.asarray(pool.x)[active]
        ys = np.asarray(pool.y)[active]
        assert np.all(xs >= 0) and np.all(xs <= WORLD_DIM)
        assert np.all(ys >= 0) and np.all(ys <= WORLD_DIM)


def test_the_agent_is_not_inside_a_wall(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        pool = board.entities
        walls = pool.type == LAVA_WALL
        overlap = (engine.collides(board.agent, pool) & walls & pool.active).any()

        assert not bool(overlap)


def test_level_seed_pins_the_level(env):
    def pinned(seed):
        return DodgeballEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))
    assert int(first.num_enemies) == int(second.num_enemies)


# --------------------------------------------------------------------------- #
# throwing
# --------------------------------------------------------------------------- #
def test_throwing_needs_the_move_action(env):
    state = floor_state(env, [], put_agent())

    board, _, _ = step_raw(env, state, THROW)
    pool = board.entities

    assert not bool((pool.active & (pool.type == PLAYER_BALL)).any()), 'standing still throws nothing'


def test_throwing_fires_along_the_last_move(env):
    """``game_step`` reads the ball direction from ``last_move_action``, not the throw."""
    state = jit_reset(jax.random.PRNGKey(0))
    board = state.env_state
    for _ in range(8):  # past the seven-step cooldown
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))

    board, _, _ = jit_raw(board, jnp.asarray(THROW))
    pool = board.entities
    ball = np.asarray(pool.active) & (np.asarray(pool.type) == PLAYER_BALL)
    index = int(np.argmax(ball))

    assert bool(ball[index]), 'the ball is thrown'
    assert float(pool.vx[index]) > 0, 'and it flies the way the agent was walking'


def test_the_throw_has_a_cooldown(env):
    """``cur_time - last_fire_time >= 7``."""
    ready = floor_state(env, [], put_agent(), last_fire_time=0, steps=6)
    board, _, _ = step_raw(env, ready, THROW)
    assert bool((board.entities.active & (board.entities.type == PLAYER_BALL)).any())

    cooling = floor_state(env, [], put_agent(), last_fire_time=1, steps=6)
    board, _, _ = step_raw(env, cooling, THROW)
    assert not bool((board.entities.active & (board.entities.type == PLAYER_BALL)).any())


# --------------------------------------------------------------------------- #
# hits
# --------------------------------------------------------------------------- #
def test_a_ball_kills_an_enemy_for_two_points(env):
    state = floor_state(env, [enemy_at(4.0, 4.0), ball_at(4.0, 4.0)], put_agent())

    board, reward, _ = step_raw(env, state)

    assert reward == pytest.approx(ENEMY_REWARD)
    assert not bool(board.entities.active[0]), 'the enemy is erased'
    assert not bool(board.entities.active[1]), 'the ball is spent'


def test_a_lava_wall_eats_the_ball(env):
    state = floor_state(env, [wall_at(4.0, 4.0), ball_at(4.0, 4.0)], put_agent())

    board, reward, _ = step_raw(env, state)

    assert not bool(board.entities.active[1]), 'the ball is destroyed'
    assert bool(board.entities.active[0]), 'the wall stays'
    assert reward == 0.0


def test_an_enemy_ball_ends_the_episode(env):
    state = floor_state(env, [ball_at(10.0, 10.0, kind=ENEMY_BALL)], put_agent())

    _, _, terminated = step_raw(env, state)

    assert terminated


def test_an_enemy_ends_the_episode_on_contact(env):
    state = floor_state(env, [enemy_at(10.0, 10.0)], put_agent())

    _, _, terminated = step_raw(env, state)

    assert terminated


def test_a_lava_wall_ends_the_episode_on_contact(env):
    state = floor_state(env, [wall_at(10.0, 10.0, 2.0, 2.0)], put_agent())

    _, _, terminated = step_raw(env, state)

    assert terminated


def test_balls_expire(env):
    """``expire_time = 50``: the engine ages projectiles away."""
    ball = engine.entity(
        10.0,
        10.0,
        rx=BALL_RADIUS,
        ry=BALL_RADIUS,
        type=PLAYER_BALL,
        image_type=PLAYER_BALL,
        collides_with_entities=True,
        expire_time=50,
        active=True,
    )

    for _ in range(50):
        ball = engine.entity_step(ball)
    assert bool(ball.active)

    for _ in range(2):
        ball = engine.entity_step(ball)
    assert not bool(ball.active)


# --------------------------------------------------------------------------- #
# the door
# --------------------------------------------------------------------------- #
def test_the_door_stays_shut_while_enemies_live(env):
    state = floor_state(env, [door_at(10.0, 10.0), enemy_at(2.0, 2.0)], put_agent(10.0, 10.0), num_enemies=1)

    board, reward, terminated = step_raw(env, state)

    assert not terminated, 'a closed door is harmless'
    assert reward == 0.0


def test_the_door_opens_when_the_last_enemy_dies(env):
    state = floor_state(env, [door_at(10.0, 10.0)], put_agent(2.0, 2.0), num_enemies=0)

    board, _, _ = step_raw(env, state)
    sprites = np.asarray(board.entities.image_type)[np.asarray(board.entities.active)]

    assert int(board.num_enemies) == 0
    assert sprites.tolist().count(DOOR_OPEN) == 1, 'the sprite switches to the open block'


def test_walking_into_an_open_door_wins(env):
    state = floor_state(env, [door_at(10.0, 10.0)], put_agent(10.0, 10.0), num_enemies=0)

    _, reward, terminated = step_raw(env, state)

    assert terminated
    assert reward == pytest.approx(COMPLETION_BONUS)


def test_the_enemy_count_tracks_the_pool(env):
    state = floor_state(env, [enemy_at(2.0, 2.0), enemy_at(4.0, 4.0)], put_agent(10.0, 10.0))

    board, _, _ = step_raw(env, state)

    assert int(board.num_enemies) == 2


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = DodgeballEnv(timeout=5)
    state = floor_state(env, [], put_agent(), steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, THROW, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
