"""ProcGen's ``ninja``: a platformer with a charged jump and throwing stars.

Rules from ``procgen/src/games/ninja.cpp``: the level wanders right over platforms
with a floor of fire and bombs on the ceilings, holding the down action charges a
jump that releases when the action does, and actions 9-12 throw a star up,
diagonally or straight ahead.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.ninja import (
    AGENT_RADIUS,
    BOMB_TILE,
    FIRE_COOLDOWN,
    FIRE_TILE,
    GOAL,
    GOAL_REWARD,
    JUMP_CHARGE_INC,
    MAX_ENTITIES,
    MAX_JUMP,
    SPACE,
    THROWING_STAR,
    TIMEOUT,
    VISIBILITY,
    WALL_MID,
    WORLD_DIM,
    NinjaEnv,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
THROW_FORWARD, THROW_DIAG, THROW_UP = 9, 10, 11
JIT_ENV = NinjaEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, grid=None, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
        steps=jnp.asarray(steps, jnp.int32),
    )
    if grid is not None:
        board = board._replace(grid=grid)
    return env.state_from(board)


def put_agent(x=10.0, y=10.5):
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
    return jnp.full((WORLD_DIM * WORLD_DIM,), SPACE, jnp.int32)


def floor_at(row, flat=None):
    flat = open_grid() if flat is None else flat
    return flat.at[row * WORLD_DIM : (row + 1) * WORLD_DIM].set(WALL_MID)


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


def test_reset_builds_a_level_with_fire_and_a_mushroom(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid)
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert grid.shape == (WORLD_DIM * WORLD_DIM,)
    assert set(np.unique(grid).tolist()) <= {WALL_MID, FIRE_TILE, BOMB_TILE, SPACE}
    assert int((grid == FIRE_TILE).sum()) > 0, 'the floor is on fire'
    assert int((grid == WALL_MID).sum()) > 500
    assert types.tolist().count(GOAL) == 1


def test_the_mushroom_sits_to_the_right(env):
    for seed in range(4):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        pool = board.entities
        goal = int(np.argmax(np.asarray(pool.type) == GOAL))

        assert float(pool.x[goal]) > float(board.agent.x) + 10


def test_the_border_is_solid(env):
    board = jit_reset(jax.random.PRNGKey(0)).env_state
    grid = np.asarray(board.grid).reshape(WORLD_DIM, WORLD_DIM)

    assert np.all(grid[0] == WALL_MID) and np.all(grid[:, 0] == WALL_MID)
    assert np.all(grid[-1] == WALL_MID) and np.all(grid[:, -1] == WALL_MID)


def test_level_seed_pins_the_level(env):
    def pinned(seed):
        return NinjaEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(2), pinned(2)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))


# --------------------------------------------------------------------------- #
# movement and the charged jump
# --------------------------------------------------------------------------- #
def test_holding_down_charges_a_jump(env):
    state = state_with(env, agent=put_agent(10.0, 11.5), grid=floor_at(10))

    board, _, _ = step_raw(env, state, DOWN)
    charged = float(board.jump_charge)

    assert charged == pytest.approx(JUMP_CHARGE_INC)
    assert bool(board.has_support)


def test_the_charge_is_capped(env):
    state = state_with(env, agent=put_agent(10.0, 11.5), grid=floor_at(10))
    board = state.env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(DOWN))

    assert float(board.jump_charge) == pytest.approx(1.0)


def test_releasing_the_action_launches(env):
    state = state_with(env, agent=put_agent(10.0, 11.5), grid=floor_at(10))
    board = state.env_state
    for _ in range(4):
        board, _, _ = jit_raw(board, jnp.asarray(DOWN))
    charge = float(board.jump_charge)

    board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    assert charge > 0
    assert float(board.agent.vy) == pytest.approx(charge * MAX_JUMP)


def test_gravity_pulls_the_zombie_down(env):
    state = state_with(env, agent=put_agent(10.0, 20.5), grid=open_grid())
    board = state.env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    assert float(board.agent.y) < 20.5


def test_the_zombie_runs_right(env):
    state = state_with(env, agent=put_agent(10.0, 11.5), grid=floor_at(10))
    board = state.env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))

    assert float(board.agent.x) > 10.0


def test_bricks_block_the_zombie(env):
    grid = floor_at(10)
    grid = grid.at[11 * WORLD_DIM + 11 : 12 * WORLD_DIM + 11 : WORLD_DIM].set(WALL_MID)
    state = state_with(env, agent=put_agent(10.0, 11.5), grid=grid)
    board = state.env_state
    for _ in range(20):
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))

    assert float(board.agent.x) < 11.0, 'the brick at column 11 holds it back'


# --------------------------------------------------------------------------- #
# stars
# --------------------------------------------------------------------------- #
def test_throwing_a_star(env):
    state = state_with(env, agent=put_agent(10.0, 10.5), grid=open_grid(), steps=FIRE_COOLDOWN)

    board, _, _ = step_raw(env, state, THROW_FORWARD)
    pool = board.entities
    stars = pool.active & (pool.type == THROWING_STAR)

    assert bool(stars.any())
    index = int(np.argmax(stars))
    assert float(pool.vx[index]) > 0, 'thrown forward'


def test_a_reflected_throw_goes_left(env):
    agent = engine.entity(
        10.0,
        10.5,
        rx=AGENT_RADIUS,
        ry=AGENT_RADIUS,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        is_reflected=True,
        render_z=1,
        smart_step=True,
    )
    state = state_with(env, agent=agent, grid=open_grid(), steps=FIRE_COOLDOWN)

    board, _, _ = step_raw(env, state, THROW_FORWARD)
    pool = board.entities
    index = int(np.argmax(np.asarray(pool.active) & (np.asarray(pool.type) == THROWING_STAR)))

    assert float(pool.vx[index]) < 0


def test_the_throw_has_a_cooldown(env):
    state = state_with(env, agent=put_agent(10.0, 10.5), grid=open_grid(), steps=FIRE_COOLDOWN)
    board = state.env_state
    board, _, _ = jit_raw(board, jnp.asarray(THROW_FORWARD))

    board, _, _ = jit_raw(board, jnp.asarray(THROW_FORWARD))
    stars = (np.asarray(board.entities.active) & (np.asarray(board.entities.type) == THROWING_STAR)).sum()

    assert int(stars) == 1, f'the cooldown is {FIRE_COOLDOWN} steps'


def test_a_star_dies_on_a_brick(env):
    grid = open_grid().at[10 * WORLD_DIM : 11 * WORLD_DIM].set(WALL_MID)
    star = engine.entity(
        10.5,
        10.5,
        0.0,
        0.0,
        rx=0.25,
        ry=0.25,
        type=THROWING_STAR,
        image_type=THROWING_STAR,
        collides_with_entities=True,
        smart_step=True,
        expire_time=15,
        active=True,
    )
    state = state_with(env, [star], put_agent(5.0, 5.5), grid=grid)

    board, _, _ = step_raw(env, state, NOOP)

    assert not bool(board.entities.active[0])


def test_a_star_clears_a_bomb(env):
    grid = open_grid()
    cell = 10 * WORLD_DIM + 10
    grid = grid.at[cell].set(BOMB_TILE)
    star = engine.entity(
        10.5,
        10.5,
        0.0,
        0.0,
        rx=0.25,
        ry=0.25,
        type=THROWING_STAR,
        image_type=THROWING_STAR,
        collides_with_entities=True,
        smart_step=True,
        expire_time=15,
        active=True,
    )
    state = state_with(env, [star], put_agent(5.0, 5.5), grid=grid)

    board, _, _ = step_raw(env, state, NOOP)

    assert not bool(board.entities.active[0]), 'the star is spent'
    assert int(board.grid[cell]) == SPACE, 'and the bomb is gone'


# --------------------------------------------------------------------------- #
# contact
# --------------------------------------------------------------------------- #
def test_standing_on_fire_ends_the_run(env):
    grid = open_grid().at[10 * WORLD_DIM : 11 * WORLD_DIM].set(FIRE_TILE)
    state = state_with(env, agent=put_agent(10.0, 10.5), grid=grid)

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


def test_the_mushroom_wins(env):
    mushroom = engine.entity(10.0, 10.5, rx=0.5, ry=0.5, type=GOAL, image_type=GOAL, image_theme=0, active=True)
    state = state_with(env, [mushroom], put_agent(10.0, 10.5), grid=open_grid())

    _, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert reward == pytest.approx(GOAL_REWARD)


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = NinjaEnv(timeout=5)
    state = state_with(env, agent=put_agent(), grid=open_grid())
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_the_view_is_sixteen_tiles(env):
    assert VISIBILITY == 16
    state = jit_reset(jax.random.PRNGKey(0))

    assert np.asarray(state.obs).shape == (64, 64, 3)


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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([RIGHT, THROW_UP, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
