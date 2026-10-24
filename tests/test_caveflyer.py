"""ProcGen's ``caveflyer``: a lunar lander in a cellular-automata cave.

Rules from ``procgen/src/games/caveflyer.cpp``: the ship thrusts along its facing
(the action's vertical axis is thrust, the horizontal one turns), red saucers take
five laser hits and pay three, and anything else you touch ends the run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.caveflyer import (
    AGENT_RADIUS,
    BULLET_RX,
    BULLET_RY,
    CAVEWALL,
    ENEMY,
    GOAL,
    GOAL_REWARD,
    MAX_ENTITIES,
    MAX_OBJECTS,
    OBSTACLE,
    PLAYER_BULLET,
    SPACE,
    TARGET,
    TARGET_HEALTH,
    TARGET_REWARD,
    TIMEOUT,
    WORLD_DIM,
    CaveFlyerEnv,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
THRUST = DOWN
FIRE = 9
JIT_ENV = CaveFlyerEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, grid=None):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
    )
    if grid is not None:
        board = board._replace(grid=grid)
    return env.state_from(board)


def put_agent(x=20.0, y=20.0, rotation=0.0):
    return engine.entity(
        x,
        y,
        rx=AGENT_RADIUS,
        ry=AGENT_RADIUS,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        rotation=rotation,
        render_z=1,
        smart_step=True,
    )


def open_grid():
    return jnp.full((WORLD_DIM * WORLD_DIM,), SPACE, jnp.int32)


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


def test_reset_carves_a_cave_with_a_saucer(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid)
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert grid.shape == (WORLD_DIM * WORLD_DIM,)
    assert set(np.unique(grid).tolist()) <= {CAVEWALL, SPACE}
    assert 100 < int((grid == SPACE).sum()) < 1500
    assert types.tolist().count(GOAL) == 1


def test_reset_scatters_three_kinds_of_object(env):
    for seed in range(4):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        types = np.asarray(pool.type)[np.asarray(pool.active)].tolist()

        assert len(types) <= 1 + MAX_OBJECTS
        assert types.count(OBSTACLE) == types.count(TARGET) == types.count(ENEMY)


def test_the_ship_starts_in_the_cave(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        index = int(board.agent.y) * WORLD_DIM + int(board.agent.x)

        assert int(board.grid[index]) == SPACE


def test_level_seed_pins_the_cave(env):
    def pinned(seed):
        return CaveFlyerEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))
    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))


# --------------------------------------------------------------------------- #
# thrust and turning
# --------------------------------------------------------------------------- #
def test_thrust_pushes_along_the_facing(env):
    state = state_with(env, agent=put_agent(20.0, 20.0, rotation=0.0), grid=open_grid())

    board, _, _ = step_raw(env, state, THRUST)

    assert float(board.agent.vy) > 0, 'rotation 0 points up: theta = pi/2'


def test_thrust_follows_a_turned_ship(env):
    state = state_with(env, agent=put_agent(20.0, 20.0, rotation=jnp.pi / 2), grid=open_grid())

    board, _, _ = step_raw(env, state, THRUST)

    # theta = -rotation + pi/2: upright the thrust is +y, a quarter turn makes it +x
    assert float(board.agent.vx) > 0.01, 'the thrust follows the ship around'


def test_the_horizontal_axis_turns_the_ship(env):
    state = state_with(env, agent=put_agent(20.0, 20.0), grid=open_grid())

    board, _, _ = step_raw(env, state, 6)

    assert float(board.agent.vrot) > 0
    assert float(board.agent.rotation) > 0, 'rotation integrates every step'


def test_the_cave_walls_stop_the_ship(env):
    grid = open_grid().at[20 * WORLD_DIM : 21 * WORLD_DIM].set(CAVEWALL)
    state = state_with(env, agent=put_agent(20.0, 19.0), grid=grid)
    board = state.env_state
    for _ in range(20):
        board, _, _ = jit_raw(board, jnp.asarray(THRUST))

    assert float(board.agent.y) < 20.0, 'the wall at row 20 holds it back'


# --------------------------------------------------------------------------- #
# lasers and contact
# --------------------------------------------------------------------------- #
def test_firing_makes_a_laser(env):
    state = state_with(env, agent=put_agent(20.0, 20.0), grid=open_grid())

    board, _, _ = step_raw(env, state, FIRE)
    pool = board.entities

    assert bool((pool.active & (pool.type == PLAYER_BULLET)).any())


def test_a_laser_dies_on_a_cave_wall(env):
    grid = open_grid().at[20 * WORLD_DIM : 21 * WORLD_DIM].set(CAVEWALL)
    laser = engine.entity(
        20.5,
        20.5,
        0.0,
        0.0,
        rx=BULLET_RX,
        ry=BULLET_RY,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        expire_time=10,
        active=True,
    )
    state = state_with(env, [laser], put_agent(5.0, 5.0), grid=grid)

    board, _, _ = step_raw(env, state, NOOP)

    assert not bool(board.entities.active[0]), 'the laser is inside the wall'


def test_five_hits_kill_a_saucer(env):
    target = engine.entity(
        20.5,
        20.5,
        rx=0.5,
        ry=0.5,
        type=TARGET,
        image_type=TARGET,
        health=TARGET_HEALTH,
        collides_with_entities=True,
        active=True,
    )
    lasers = [
        engine.entity(
            20.5,
            20.5,
            0.0,
            0.0,
            rx=BULLET_RX,
            ry=BULLET_RY,
            type=PLAYER_BULLET,
            image_type=PLAYER_BULLET,
            expire_time=10,
            active=True,
        )
        for _ in range(int(TARGET_HEALTH))
    ]
    state = state_with(env, [target, *lasers], put_agent(5.0, 5.0), grid=open_grid())

    board, reward, _ = step_raw(env, state, NOOP)

    assert not bool(board.entities.active[0]), 'the saucer is destroyed'
    assert reward == pytest.approx(TARGET_REWARD)


def test_reaching_the_saucer_wins(env):
    saucer = engine.entity(
        20.5, 20.5, rx=0.5, ry=0.5, type=GOAL, image_type=GOAL, collides_with_entities=True, active=True
    )
    state = state_with(env, [saucer], put_agent(20.5, 20.5), grid=open_grid())

    _, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert reward == pytest.approx(GOAL_REWARD)


def test_touching_an_obstacle_ends_the_run(env):
    rock = engine.entity(
        20.5, 20.5, rx=0.5, ry=0.5, type=OBSTACLE, image_type=OBSTACLE, collides_with_entities=True, active=True
    )
    state = state_with(env, [rock], put_agent(20.5, 20.5), grid=open_grid())

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = CaveFlyerEnv(timeout=5)
    state = state_with(env, agent=put_agent(), grid=open_grid())
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_observation_shows_the_cave(env):
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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, THRUST, FIRE]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
