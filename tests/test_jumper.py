"""ProcGen's ``jumper``: a cave platformer with a double jump and a carrot.

Rules from ``procgen/src/games/jumper.cpp``: the level is the largest room of a
roughened maze, carved back open along the path to the carrot; the rabbit jumps
twice before it must land again, and touching a spike ends the run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.jumper import (
    AGENT_RX,
    AGENT_RY,
    CAVEWALL,
    CAVEWALL_TOP,
    GOAL,
    GOAL_REWARD,
    JUMPS_PER_LANDING,
    MAX_ENTITIES,
    SPACE,
    SPIKE,
    TIMEOUT,
    WORLD_DIM,
    JumperEnv,
    empty_pool,
)

UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
JIT_ENV = JumperEnv()
jit_step = jax.jit(JIT_ENV.step)
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)
jit_raw = jax.jit(JIT_ENV.step_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, entities=(), agent=None, flat=None):
    base = jit_reset_state(jax.random.PRNGKey(0))
    board = base._replace(
        agent=base.agent if agent is None else agent,
        entities=make_pool(entities),
    )
    if flat is not None:
        board = board._replace(grid=flat)
    return env.state_from(board)


def open_flat():
    return jnp.full((WORLD_DIM * WORLD_DIM,), SPACE, jnp.int32)


def floor_row(row, flat=None):
    """A solid cave floor under ``row`` so the rabbit has something to stand on."""
    flat = open_flat() if flat is None else flat
    return flat.at[row * WORLD_DIM : (row + 1) * WORLD_DIM].set(CAVEWALL)


def put_agent(x=20.0, y=10.5):
    flat = floor_row(10)
    return (
        engine.entity(
            x,
            y,
            rx=AGENT_RX,
            ry=AGENT_RY,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        ),
        flat,
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


def test_reset_builds_a_cave_with_a_carrot(env):
    state = jit_reset(jax.random.PRNGKey(0))
    flat = np.asarray(state.env_state.grid)
    pool = state.env_state.entities
    active = np.asarray(pool.active)
    types = np.asarray(pool.type)[active]

    assert flat.shape == (WORLD_DIM * WORLD_DIM,)
    assert set(np.unique(flat).tolist()) <= {CAVEWALL, CAVEWALL_TOP, SPACE}
    assert types.tolist().count(GOAL) == 1, 'one carrot per level'
    assert 100 < int((flat == SPACE).sum()) < 1200, 'the playable floor is carved out'


def test_reset_scatters_a_few_spikes(env):
    for seed in range(5):
        state = jit_reset(jax.random.PRNGKey(seed))
        pool = state.env_state.entities
        active = np.asarray(pool.active)
        types = np.asarray(pool.type)[active].tolist()

        assert 0 <= types.count(SPIKE) <= 60
        assert int(active.sum()) == 1 + types.count(SPIKE)


def test_the_rabbit_starts_on_the_floor(env):
    for seed in range(5):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        flat = np.asarray(board.grid).reshape(WORLD_DIM, WORLD_DIM)
        column = int(board.agent.x)
        row = int(board.agent.y)

        assert flat[row, column] == SPACE
        assert flat[row - 1, column] in (CAVEWALL, CAVEWALL_TOP, 51), 'something to stand on'


def test_the_wall_theme_is_one_of_four(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert 0 <= int(state.env_state.wall_theme) < 4


def test_level_seed_pins_the_level(env):
    def pinned(seed):
        return JumperEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state

    first, second = pinned(4), pinned(4)

    assert np.array_equal(np.asarray(first.grid), np.asarray(second.grid))
    assert np.array_equal(np.asarray(first.entities.x), np.asarray(second.entities.x))


# --------------------------------------------------------------------------- #
# jumping and gravity
# --------------------------------------------------------------------------- #
def test_jumping_lifts_the_rabbit(env):
    agent, flat = put_agent()
    state = state_with(env, agent=agent, flat=flat)
    start = float(state.env_state.agent.y)

    board, _, _ = step_raw(env, state, DOWN)

    assert float(board.agent.y) > start


def test_the_rabbit_gets_two_jumps_per_landing(env):
    """``jump_count`` refills on support, and the jump cooldown needs four steps."""
    agent, flat = put_agent()
    board = state_with(env, agent=agent, flat=flat).env_state
    for _ in range(4):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    board, _, _ = jit_raw(board, jnp.asarray(DOWN))

    assert int(board.jump_count) == JUMPS_PER_LANDING - 1


def test_jumping_needs_something_to_stand_on(env):
    agent = engine.entity(
        20.0,
        10.5,
        rx=AGENT_RX,
        ry=AGENT_RY,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        render_z=1,
        smart_step=True,
    )
    state = state_with(env, agent=agent, flat=open_flat())

    board, _, _ = step_raw(env, state, DOWN)

    assert not bool(board.has_support), 'nothing underfoot'
    assert int(board.jump_count) == 0, 'so the jump is refused'


def test_gravity_pulls_the_rabbit_down(env):
    agent = engine.entity(
        20.0,
        20.5,
        rx=AGENT_RX,
        ry=AGENT_RY,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        render_z=1,
        smart_step=True,
    )
    board = state_with(env, agent=agent, flat=open_flat()).env_state
    for _ in range(10):
        board, _, _ = jit_raw(board, jnp.asarray(NOOP))

    assert float(board.agent.y) < 20.5


def test_the_rabbit_runs_sideways(env):
    agent, flat = put_agent()
    board = state_with(env, agent=agent, flat=flat).env_state
    for _ in range(8):
        board, _, _ = jit_raw(board, jnp.asarray(RIGHT))
    right = float(board.agent.x)
    for _ in range(8):
        board, _, _ = jit_raw(board, jnp.asarray(LEFT))

    assert right > 20.0
    assert float(board.agent.x) < right


def test_the_sprite_tracks_the_direction(env):
    agent, flat = put_agent()
    board = state_with(env, agent=agent, flat=flat).env_state

    board, _, _ = jit_raw(board, jnp.asarray(RIGHT))
    assert bool(board.facing_right)
    board, _, _ = jit_raw(board, jnp.asarray(LEFT))

    assert not bool(board.facing_right)


# --------------------------------------------------------------------------- #
# contact
# --------------------------------------------------------------------------- #
def test_the_carrot_wins_the_level(env):
    agent, flat = put_agent()
    carrot = engine.entity(agent.x, agent.y, rx=0.5, ry=0.5, type=GOAL, image_type=GOAL, active=True)

    board, reward, terminated = step_raw(env, state_with(env, [carrot], agent, flat))

    assert terminated
    assert reward == pytest.approx(GOAL_REWARD)


def test_a_spike_ends_the_run(env):
    agent = engine.entity(
        20.0,
        20.5,
        rx=AGENT_RX,
        ry=AGENT_RY,
        type=engine.PLAYER,
        image_type=engine.PLAYER,
        render_z=1,
        smart_step=True,
    )
    spike = engine.entity(agent.x, agent.y, rx=0.23, ry=0.4, type=SPIKE, image_type=SPIKE, active=True)

    _, reward, terminated = step_raw(env, state_with(env, [spike], agent, open_flat()))

    assert terminated
    assert reward == 0.0


# --------------------------------------------------------------------------- #
# episodes and observations
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = JumperEnv(timeout=5)
    agent, flat = put_agent()
    state = state_with(env, agent=agent, flat=flat)
    state = env.state_from(state.env_state._replace(steps=jnp.asarray(4, jnp.int32)))

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])


def test_timeout_matches_the_original():
    assert TIMEOUT == 1000


def test_the_camera_follows_the_rabbit(env):
    """A wall near one rabbit and not the other, so the frames must differ."""
    low_flat = open_flat().at[10 * WORLD_DIM : 11 * WORLD_DIM].set(CAVEWALL)
    high_flat = open_flat()
    agent, _ = put_agent(10.0, 10.5)
    low = state_with(env, agent=agent, flat=low_flat)
    agent, _ = put_agent(30.0, 10.5)
    high = state_with(env, agent=agent, flat=high_flat)

    assert not np.array_equal(np.asarray(low.obs), np.asarray(high.obs))


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

    stepped = jax.vmap(jax.jit(env.step))(states, jnp.array([UP, DOWN, NOOP]))

    assert stepped.obs.shape == (3, 64, 64, 3)
    assert stepped.reward.shape == (3,)
