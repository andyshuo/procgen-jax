"""ProcGen's ``heist``: a locked maze, one gem, one key per door.

Rules from ``procgen/src/games/heist.cpp``: a cell of the maze is the start, every
door needs its own key before the agent can pass, and reaching the gem pays ten
completion points (assigned, not added).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.heist import (
    AGENT_RADIUS,
    COMPLETION_BONUS,
    DOOR_RADIUS,
    EXIT,
    KEY,
    LOCKED_DOOR,
    MAX_ENTITIES,
    MAX_KEYS,
    TIMEOUT,
    WALL_TILE,
    WORLD_DIM,
    HeistEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

# set_action_xy: vx = act // 3 - 1, vy = act % 3 - 1, so 3 is up and 5 is down.
UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
JIT_ENV = HeistEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool()
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, has_keys=None, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            has_keys=jnp.zeros((MAX_KEYS,), bool) if has_keys is None else jnp.asarray(has_keys),
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def open_grid():
    return jnp.zeros((WORLD_DIM, WORLD_DIM), jnp.int32)


def heist_state(env, grid, entities, agent, has_keys=None, steps=0):
    """A state with an explicit grid, so the maze does not interfere."""
    state = state_with(env, entities=entities, agent=agent, has_keys=has_keys, steps=steps)
    return env.state_from(state.env_state._replace(grid=grid))


def put_agent(x, y):
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


def door_at(x, y, theme=0):
    return engine.entity(
        x,
        y,
        rx=DOOR_RADIUS,
        ry=DOOR_RADIUS,
        type=LOCKED_DOOR,
        image_type=LOCKED_DOOR,
        image_theme=theme,
        active=True,
    )


def key_at(x, y, theme=0):
    return engine.entity(
        x, y, rx=AGENT_RADIUS, ry=AGENT_RADIUS, type=KEY, image_type=KEY, image_theme=theme, active=True
    )


def gem_at(x, y):
    return engine.entity(x, y, rx=AGENT_RADIUS, ry=AGENT_RADIUS, type=EXIT, image_type=EXIT, active=True)


@pytest.fixture(scope='module')
def env():
    return JIT_ENV


# --------------------------------------------------------------------------- #
# interface and level generation
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)
    assert env.observation_size == 64 * 64 * 3


def test_reset_builds_a_maze_with_one_gem(env):
    state = jit_reset(jax.random.PRNGKey(0))
    grid, pool = state.env_state.grid, state.env_state.entities

    assert grid.shape == (WORLD_DIM, WORLD_DIM)
    assert set(np.unique(grid).tolist()) <= {SPACE, WALL_TILE}
    types = np.asarray(pool.type)[np.asarray(pool.active)]
    assert types.tolist().count(EXIT) == 1, 'exactly one gem per level'


def test_every_door_has_its_key(env):
    for seed in range(6):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        active = np.asarray(pool.active)
        types = np.asarray(pool.type)[active]
        themes = np.asarray(pool.image_theme)[active]

        doors = sorted(themes[types == LOCKED_DOOR].tolist())
        keys = sorted(themes[types == KEY].tolist())
        assert doors == keys, f'seed {seed}: doors {doors} vs keys {keys}'
        assert len(doors) <= MAX_KEYS


def test_the_bodies_fit_the_pool(env):
    for seed in range(6):
        state = jit_reset(jax.random.PRNGKey(seed))
        active = np.asarray(state.env_state.entities.active)

        assert int(active.sum()) <= MAX_ENTITIES
        assert int(active.sum()) >= 1


def test_the_level_is_deterministic(env):
    first = jit_reset(jax.random.PRNGKey(5)).env_state
    second = jit_reset(jax.random.PRNGKey(5)).env_state

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))
    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))


def test_level_seed_pins_the_maze(env):
    def pinned(seed):
        return HeistEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(3), pinned(3)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))


def test_the_agent_starts_on_a_floor_cell(env):
    for seed in range(6):
        state = jit_reset(jax.random.PRNGKey(seed))
        board = state.env_state
        cell = board.grid[int(board.agent.y), int(board.agent.x)]

        assert int(cell) == SPACE


def test_the_bodies_do_not_overlap(env):
    for seed in range(6):
        pool = jit_reset(jax.random.PRNGKey(seed)).env_state.entities
        active = np.asarray(pool.active)

        pairs = jax.vmap(lambda a, p=pool: jax.vmap(lambda b: engine.collides(a, b))(p))(pool)
        overlap = np.asarray(pairs)[active][:, active]
        np.fill_diagonal(overlap, False)
        assert not overlap.any(), f'seed {seed} has overlapping bodies'


def test_bad_maze_dim_is_rejected():
    with pytest.raises(ValueError):
        HeistEnv(maze_dim=8)
    with pytest.raises(ValueError):
        HeistEnv(maze_dim=3)


# --------------------------------------------------------------------------- #
# movement and doors
# --------------------------------------------------------------------------- #
def test_the_agent_only_moves_horizontally_when_pushed_sideways(env):
    state = heist_state(env, open_grid(), [], put_agent(6.5, 6.5))
    start_y = float(state.env_state.agent.y)

    state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.y) == start_y
    assert float(state.env_state.agent.x) > 6.5


def test_a_wall_blocks_the_agent(env):
    _ = jnp.pad(jnp.ones((12, 13), jnp.int32), ((0, 0), (0, 0)))[:WORLD_DIM]
    grid = jnp.ones((WORLD_DIM, WORLD_DIM), jnp.int32).at[6, :].set(WALL_TILE)
    state = heist_state(env, grid, [], put_agent(6.5, 5.5))

    for _ in range(30):
        state = jit_step(state, jnp.asarray(DOWN))

    assert float(state.env_state.agent.y) < 6.0, 'the wall at row 6 stops the agent'
    assert int(state.env_state.grid[6, 6]) == WALL_TILE


def test_a_door_blocks_until_its_key_is_held(env):
    state = heist_state(env, open_grid(), [door_at(6.5, 6.5)], put_agent(5.0, 6.5))

    for _ in range(30):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.x) < 6.5, 'a locked door is solid'
    assert bool(state.env_state.entities.active[0]), 'the door is still there'


def test_a_key_opens_the_door(env):
    state = heist_state(
        env,
        open_grid(),
        [door_at(6.5, 6.5)],
        put_agent(5.0, 6.5),
        has_keys=[True, False, False],
    )

    for _ in range(40):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.x) > 6.5, 'the agent walks through the door'
    assert not bool(state.env_state.entities.active[0]), 'the door opens and is removed'


def test_walking_over_a_key_adds_it_to_the_ring(env):
    state = heist_state(env, open_grid(), [key_at(6.5, 6.5, theme=1)], put_agent(5.5, 6.5))

    for _ in range(10):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert bool(state.env_state.has_keys[1])
    assert not bool(state.env_state.has_keys[0])
    assert not bool(state.env_state.entities.active[0]), 'the key is picked up'


def test_the_gem_pays_the_bonus_and_ends_the_level(env):
    state = heist_state(env, open_grid(), [gem_at(6.5, 6.5)], put_agent(6.5, 6.5))

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == COMPLETION_BONUS
    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])
    assert not bool(stepped.info['truncated'])


def test_the_bonus_is_assigned_not_added(env):
    state = heist_state(env, open_grid(), [gem_at(6.5, 6.5)], put_agent(6.5, 6.5))

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == COMPLETION_BONUS, 'the original assigns reward = 10'


def test_ordinary_steps_pay_nothing(env):
    state = heist_state(env, open_grid(), [key_at(10.0, 2.0)], put_agent(6.5, 6.5))

    stepped = jit_step(state, jnp.asarray(LEFT))

    assert float(stepped.reward) == 0.0
    assert not bool(stepped.done)


def test_the_agent_faces_the_way_it_walks(env):
    state = heist_state(env, open_grid(), [], put_agent(6.5, 6.5))

    right = jit_step(state, jnp.asarray(RIGHT))
    left = jit_step(right, jnp.asarray(LEFT))

    assert not bool(right.env_state.agent.is_reflected)
    assert bool(left.env_state.agent.is_reflected)


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = HeistEnv(timeout=5)
    state = heist_state(env, open_grid(), [], put_agent(6.5, 6.5), steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])
    assert int(stepped.env_state.steps) == 0


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_auto_reset_gives_a_new_level(env):
    state = heist_state(env, open_grid(), [gem_at(6.5, 6.5)], put_agent(6.5, 6.5))

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done), 'the transition that ended the level stays done'
    assert int(stepped.env_state.steps) == 0, 'but the state is a fresh episode'
    assert int((np.asarray(stepped.env_state.grid) == WALL_TILE).sum()) > 0


def test_a_whole_episode_runs_under_scan(env):
    state = jit_reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (128,), 0, 15)

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, carry.reward

    final_state, rewards = jax.jit(lambda: jax.lax.scan(body, state, actions))()

    assert rewards.shape == (128,)
    assert final_state.obs.shape == (64, 64, 3)


# --------------------------------------------------------------------------- #
# observations, jit, vmap
# --------------------------------------------------------------------------- #
def test_observation_shows_the_maze(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert len({tuple(pixel) for pixel in obs.reshape(-1, 3)}) > 5


def test_step_is_jittable(env):
    state = jit_reset(jax.random.PRNGKey(0))

    stepped = jax.jit(env.step)(state, jnp.asarray(RIGHT))

    assert stepped.obs.shape == state.obs.shape
    assert jnp.isfinite(stepped.reward)


def test_vmap_reset_and_step(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    states = jax.vmap(jax.jit(env.reset))(keys)
    assert states.obs.shape == (3, 64, 64, 3)

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, LEFT, DOWN]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
