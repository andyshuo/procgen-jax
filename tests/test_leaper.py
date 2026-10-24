"""ProcGen's ``leaper``: hopping, traffic, logs and the finish line.

Rules from ``procgen/src/games/leaper.cpp``.  Note the action encoding: in ProcGen
``+vy`` points up the board, so action 5 hops towards the goal and action 3 hops
back towards the start.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.leaper import (
    CAR,
    GOAL_REWARD,
    LOG,
    MAX_ENTITIES,
    MAX_SPEED,
    NSTEP,
    ROAD,
    TIMEOUT,
    WATER,
    WORLD_DIM,
    LeaperEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 5, 1, 3, 4, 7
JIT_ENV = LeaperEnv()
jit_step = jax.jit(JIT_ENV.step)

# Reset runs the level generator (and, for leaper, a 300 step warm-up);
# jitting it keeps the suite out of eager scans.
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_pool()
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, agent=None, entities=(), finish=None, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            finish=base.finish if finish is None else finish,
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def body_at(x, y, kind, vx=0.0, rx=0.5, ry=0.5, theme=0, reflected=False):
    return engine.entity(
        x,
        y,
        vx,
        0.0,
        rx=rx,
        ry=ry,
        type=kind,
        image_type=kind,
        image_theme=theme,
        is_reflected=reflected,
        active=True,
    )


@pytest.fixture(scope='module')
def env():
    return LeaperEnv()


# --------------------------------------------------------------------------- #
# interface and the board
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)
    assert env.observation_size == 64 * 64 * 3


def test_reset_builds_a_board(env):
    state = jit_reset(jax.random.PRNGKey(0))
    board = state.env_state
    grid = np.asarray(board.grid)

    assert grid.shape == (WORLD_DIM, WORLD_DIM)
    assert set(np.unique(grid)) <= {SPACE, ROAD, WATER}
    road_rows = {row for row in range(WORLD_DIM) if (grid[row] == ROAD).any()}
    water_rows = {row for row in range(WORLD_DIM) if (grid[row] == WATER).any()}
    assert len(road_rows) == int(board.num_road_lanes)
    assert len(water_rows) == int(board.num_water_lanes)
    # lanes are contiguous, road below water, finish above the water
    assert road_rows == set(range(int(board.bottom_road_y), int(board.bottom_road_y) + len(road_rows)))
    assert water_rows == set(range(int(board.bottom_water_y), int(board.bottom_water_y) + len(water_rows)))
    assert int(board.goal_y) > max(water_rows)


def test_the_frog_starts_on_the_bottom_row(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert np.isclose(float(state.env_state.agent.y), float(state.env_state.agent.ry))
    assert float(state.env_state.agent.x) > 0


def test_reset_warms_the_traffic_up(env):
    """The original runs 300 spawn-and-step iterations before the agent starts."""
    state = jit_reset(jax.random.PRNGKey(0))
    pool = state.env_state.entities

    assert int(pool.active.sum()) > 0
    assert int((pool.active & (pool.type == LOG)).sum()) + int((pool.active & (pool.type == CAR)).sum()) == int(
        pool.active.sum()
    )


def test_level_seed_pins_the_board(env):
    first = LeaperEnv(level_seed=11).reset(jax.random.PRNGKey(0))
    second = LeaperEnv(level_seed=11).reset(jax.random.PRNGKey(99))

    assert np.array_equal(np.asarray(first.env_state.grid), np.asarray(second.env_state.grid))
    assert int(first.env_state.bottom_water_y) == int(second.env_state.bottom_water_y)


def test_levels_differ_across_seeds(env):
    boards = {
        np.asarray(LeaperEnv(level_seed=seed).reset(jax.random.PRNGKey(0)).env_state.grid).tobytes()
        for seed in range(6)
    }

    assert len(boards) > 1


# --------------------------------------------------------------------------- #
# hopping
# --------------------------------------------------------------------------- #
def test_a_hop_moves_exactly_one_tile(env):
    """MAX_SPEED then VEL_DECAY: 0.4 + 0.3 + 0.2 + 0.1 = 1.0 over NSTEP steps."""
    state = jit_reset(jax.random.PRNGKey(0))
    start = float(state.env_state.agent.y)

    state = jit_step(state, jnp.asarray(UP))
    for _ in range(NSTEP):
        state = jit_step(state, jnp.asarray(NOOP))

    assert np.isclose(float(state.env_state.agent.y) - start, 1.0, atol=1e-5)
    assert float(state.env_state.agent.vy) == 0.0


def test_a_hop_cannot_be_restarted_mid_air(env):
    """``update_agent_velocity`` only fires while both velocities are zero."""
    state = jit_reset(jax.random.PRNGKey(0))
    state = jit_step(state, jnp.asarray(UP))
    moving = float(state.env_state.agent.vy)

    state = jit_step(state, jnp.asarray(UP))

    assert 0 < float(state.env_state.agent.vy) < moving


def test_the_frog_animates_while_hopping(env):
    state = jit_reset(jax.random.PRNGKey(0))

    state = jit_step(state, jnp.asarray(UP))

    assert int(state.env_state.agent.image_theme) >= 1


def test_velocity_decays_to_a_stop(env):
    state = jit_reset(jax.random.PRNGKey(0))
    state = jit_step(state, jnp.asarray(RIGHT))

    for _ in range(NSTEP):
        state = jit_step(state, jnp.asarray(NOOP))

    assert float(state.env_state.agent.vx) == 0.0


@pytest.mark.parametrize('action', [9, 10, 11, 12, 13, 14])
def test_action_slots_past_the_grid_are_noops(env, action):
    state = jit_reset(jax.random.PRNGKey(0))

    stepped = jit_step(state, jnp.asarray(action))

    assert float(stepped.env_state.agent.vx) == 0.0
    assert float(stepped.env_state.agent.vy) == 0.0


# --------------------------------------------------------------------------- #
# hazards
# --------------------------------------------------------------------------- #
def test_a_car_ends_the_episode(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    car = body_at(float(base.agent.x), float(base.agent.y), CAR, rx=0.5, ry=0.25)
    state = state_with(env, entities=[car])

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])


def test_standing_still_on_water_drowns(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    agent = base.agent._replace(y=jnp.asarray(float(base.bottom_water_y) + 0.5, jnp.float32))
    state = state_with(env, agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)


def test_a_log_underneath_keeps_the_frog_alive_and_carries_it(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    agent = base.agent._replace(
        x=jnp.asarray(5.0, jnp.float32), y=jnp.asarray(float(base.bottom_water_y) + 0.5, jnp.float32)
    )
    log = body_at(5.0, float(base.bottom_water_y) + 0.5, LOG, vx=0.09, rx=0.45, ry=0.45)
    state = state_with(env, agent=agent, entities=[log])

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert not bool(stepped.done), 'a log keeps the frog out of the water'
    assert float(stepped.env_state.agent.x) > 5.0, 'and drags it along'


def test_the_finish_line_requires_standing_still(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    finish_y = float(base.finish.y)
    agent = base.agent._replace(x=jnp.asarray(float(base.finish.x), jnp.float32), y=jnp.asarray(finish_y, jnp.float32))

    still = jit_step(state_with(env, agent=agent), jnp.asarray(NOOP))
    moving = jit_step(state_with(env, agent=agent._replace(vx=jnp.asarray(MAX_SPEED, jnp.float32))), jnp.asarray(NOOP))

    assert bool(still.done)
    assert float(still.reward) == GOAL_REWARD
    assert not bool(moving.done), 'a frog that is still sliding has not arrived'


def test_timeout_truncates():
    env = LeaperEnv(timeout=5)
    state = state_with(env, steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])
    assert int(stepped.env_state.steps) == 0


def test_timeout_matches_the_original():
    assert TIMEOUT == 500


# --------------------------------------------------------------------------- #
# traffic
# --------------------------------------------------------------------------- #
def test_traffic_stays_within_the_pool(env):
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        return carry, carry.env_state.entities.active.sum()

    _, alive = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((200,), NOOP)))(state)

    assert int(alive.max()) < MAX_ENTITIES, 'the pool must not overflow'
    assert int(alive.max()) > 0


def test_traffic_drives_across_the_board(env):
    """Cars and logs only ever move horizontally, at their lane speed."""
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        pool = carry.env_state.entities
        return carry, (pool.vx, pool.vy, pool.active)

    _, (vxs, vys, active) = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((40,), NOOP)))(state)
    moving = np.asarray(vxs)[np.asarray(active)]

    assert moving.size > 0
    assert np.all(np.abs(moving) <= 0.2 + 1e-6), 'lane speeds stay in range'
    assert np.all(np.asarray(vys)[np.asarray(active)] == 0.0), 'traffic never moves vertically'


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
def test_observation_shows_the_whole_board(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert obs.shape == (64, 64, 3)
    # the road row must be drawn somewhere, and it is not the background
    assert len({tuple(pixel) for pixel in obs.reshape(-1, 3)}) > 5


def test_step_is_jittable(env):
    state = jit_reset(jax.random.PRNGKey(0))

    stepped = jax.jit(env.step)(state, jnp.asarray(UP))

    assert stepped.obs.shape == state.obs.shape
    assert jnp.isfinite(stepped.reward)


def test_vmap_reset_and_step(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = jax.vmap(env.reset)(keys)
    assert states.obs.shape == (4, 64, 64, 3)

    stepped = jax.vmap(env.step)(states, jnp.array([UP, LEFT, DOWN, NOOP]))

    assert stepped.obs.shape == (4, 64, 64, 3)
    assert stepped.reward.shape == (4,)


def test_whole_episode_runs_under_scan(env):
    state = jit_reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 15)

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, carry.reward

    final_state, rewards = jax.jit(lambda: jax.lax.scan(body, state, actions))()

    assert rewards.shape == (64,)
    assert final_state.env_state.grid.shape == (WORLD_DIM, WORLD_DIM)
