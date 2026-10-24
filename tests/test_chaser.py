"""ProcGen's ``chaser``: an orb-filled maze, one crystal per quadrant, hungry ghosts.

Rules from ``procgen/src/games/chaser.cpp``: the level is a no-dead-ends maze where
every open cell holds an orb, three eggs hatch into ghosts that walk the maze toward
(or away from) you, and eating a crystal makes the ghosts edible for 75 steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.chaser import (
    AGENT_RADIUS,
    COMPLETION_BONUS,
    EAT_TIMEOUT,
    EGG_TIMEOUT,
    ENEMY,
    ENEMY_EGG,
    ENEMY_RADIUS,
    LARGE_ORB,
    MAX_ENTITIES,
    MAZE_DIM,
    MAZE_WALL,
    ORB_REWARD,
    ORB_TILE,
    TIMEOUT,
    TOTAL_ENEMIES,
    ChaserEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
JIT_ENV = ChaserEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, grid=None, eat_time=0, collected=0, total=1):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
        eat_time=jnp.asarray(eat_time, jnp.int32),
        orbs_collected=jnp.asarray(collected, jnp.int32),
        total_orbs=jnp.asarray(total, jnp.int32),
    )
    if grid is not None:
        board = board._replace(grid=grid)
    return env.state_from(board)


def put_agent(x=6.5, y=6.5):
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
    return jnp.full((MAZE_DIM * MAZE_DIM,), SPACE, jnp.int32)


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


def test_reset_fills_a_maze_with_orbs(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid = np.asarray(state.env_state.grid)

    assert grid.shape == (MAZE_DIM * MAZE_DIM,)
    assert set(np.unique(grid).tolist()) <= {MAZE_WALL, SPACE, ORB_TILE}
    assert int((grid == ORB_TILE).sum()) > 50, 'most open cells hold an orb'
    assert int(state.env_state.total_orbs) == int((grid == ORB_TILE).sum())


def test_reset_hides_a_crystal_in_three_quadrants(env):
    for seed in range(5):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        active = np.asarray(pool.active)
        types = np.asarray(pool.type)[active].tolist()

        assert types.count(LARGE_ORB) in (3, 4), 'one crystal per quadrant, minus the drawn one'


def test_reset_spawns_three_eggs(env):
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities
    types = np.asarray(pool.type)[np.asarray(pool.active)].tolist()

    assert types.count(ENEMY_EGG) == TOTAL_ENEMIES


def test_the_eye_starts_on_an_orb(env):
    for seed in range(5):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        index = int(board.agent.y) * MAZE_DIM + int(board.agent.x)

        assert int(board.grid[index]) == ORB_TILE, 'the start cell is an orb too'


def test_level_seed_pins_the_maze(env):
    def pinned(seed):
        return ChaserEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(6), pinned(6)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))


# --------------------------------------------------------------------------- #
# orbs and crystals
# --------------------------------------------------------------------------- #
def test_standing_on_an_orb_collects_it(env):
    grid = open_grid().at[6 * MAZE_DIM + 6].set(ORB_TILE)
    state = state_with(env, agent=put_agent(6.5, 6.5), grid=grid, collected=0, total=10)

    board, reward, _ = step_raw(env, state, NOOP)

    assert int(board.orbs_collected) == 1
    assert reward == pytest.approx(ORB_REWARD)
    assert int(board.grid[6 * MAZE_DIM + 6]) == SPACE


def test_collecting_the_last_orb_wins(env):
    grid = open_grid().at[6 * MAZE_DIM + 6].set(ORB_TILE)
    state = state_with(env, agent=put_agent(6.5, 6.5), grid=grid, collected=9, total=10)

    _, reward, terminated = step_raw(env, state, NOOP)

    assert terminated
    assert reward == pytest.approx(ORB_REWARD + COMPLETION_BONUS)


def test_a_crystal_makes_the_ghosts_edible(env):
    crystal = engine.entity(6.5, 6.5, rx=0.4, ry=0.4, type=LARGE_ORB, image_type=LARGE_ORB, active=True)
    state = state_with(env, [crystal], put_agent(6.5, 6.5), eat_time=-EAT_TIMEOUT, collected=0, total=10)

    board, reward, terminated = step_raw(env, state, NOOP)

    assert int(board.eat_time) > 0
    assert reward >= ORB_REWARD - 1e-6, 'the crystal, and any orb underfoot'
    assert not bool(board.entities.active[0])
    assert not terminated


# --------------------------------------------------------------------------- #
# ghosts
# --------------------------------------------------------------------------- #
def test_a_ghost_kills_when_it_is_dangerous(env):
    ghost = engine.entity(
        6.5,
        6.5,
        rx=ENEMY_RADIUS,
        ry=ENEMY_RADIUS,
        type=ENEMY,
        image_type=ENEMY,
        smart_step=True,
        active=True,
    )
    state = state_with(env, [ghost], put_agent(6.5, 6.5), eat_time=-EAT_TIMEOUT, collected=0, total=10)

    _, _, terminated = step_raw(env, state, NOOP)

    assert terminated


def test_an_edible_ghost_is_eaten(env):
    ghost = engine.entity(
        6.5,
        6.5,
        rx=ENEMY_RADIUS,
        ry=ENEMY_RADIUS,
        type=ENEMY,
        image_type=ENEMY,
        smart_step=True,
        active=True,
    )
    state = state_with(env, [ghost], put_agent(6.5, 6.5), eat_time=0, collected=0, total=10)

    board, reward, terminated = step_raw(env, state, NOOP)

    assert not terminated
    assert not bool(board.entities.active[0]), 'the ghost is swallowed'
    assert reward >= ORB_REWARD


def test_eggs_hatch_after_their_timer(env):
    egg = engine.entity(
        6.5,
        8.5,
        rx=ENEMY_RADIUS,
        ry=ENEMY_RADIUS,
        type=ENEMY_EGG,
        image_type=ENEMY_EGG,
        health=float(EGG_TIMEOUT),
        active=True,
    )
    state = state_with(env, [egg], put_agent(6.5, 6.5), collected=0, total=10)
    board = state.env_state
    for _ in range(EGG_TIMEOUT):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    types = np.asarray(board.entities.type)[np.asarray(board.entities.active)].tolist()

    assert ENEMY in types, 'the egg became a ghost'


def test_the_maze_walls_block_the_eye(env):
    grid = open_grid().at[6 * MAZE_DIM : 7 * MAZE_DIM].set(MAZE_WALL)
    state = state_with(env, agent=put_agent(6.5, 5.5), grid=grid, collected=0, total=10)
    board = state.env_state
    for _ in range(20):
        board, _, _ = jit_raw(board, jnp.asarray(DOWN))

    assert float(board.agent.y) < 6.5, 'the wall at row 6 stops it'


def test_the_eye_snaps_to_full_speed(env):
    state = state_with(env, agent=put_agent(6.5, 6.5), grid=open_grid(), collected=0, total=10)

    board, _, _ = step_raw(env, state, DOWN)

    assert float(board.agent.vy) == pytest.approx(0.5), 'update_agent_velocity snaps to maxspeed'


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = ChaserEnv(timeout=5)
    state = state_with(env, agent=put_agent(), grid=open_grid(), collected=0, total=10)
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_observation_shows_the_maze(env):
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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, LEFT, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
