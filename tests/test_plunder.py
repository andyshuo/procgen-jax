"""ProcGen's ``plunder``: a naval duel with target identification.

Rules from ``procgen/src/games/plunder.cpp``: ships drift along five lanes, you
slide left and right, and action 9 fires a cannonball on a three-step cooldown.
Hitting the current target scores +1 and refills juice; hitting anything else
drains it; empty juice ends the episode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.plunder import (
    BULLET_EXPIRE,
    COMPLETION_BONUS,
    FIRE_ACTION,
    MAX_ENTITIES,
    NUM_LANES,
    NUM_SHIP_TYPES,
    PANEL,
    PLAYER_BULLET,
    POSITIVE_REWARD,
    SHIP,
    TARGET_BACKGROUND,
    TARGET_QUOTA,
    TIMEOUT,
    WORLD_DIM,
    PlunderEnv,
    empty_pool,
)
from procgen_jax.objects import SPACE

UP, LEFT, DOWN, NOOP, RIGHT = 5, 1, 3, 4, 7
JIT_ENV = PlunderEnv()
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


def state_with(env, entities=(), agent=None, juice=None, targets_hit=0, steps=0):
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            entities=make_pool(entities),
            juice_left=base.juice_left if juice is None else jnp.asarray(juice, jnp.float32),
            targets_hit=jnp.asarray(targets_hit, jnp.int32),
            steps=jnp.asarray(steps, jnp.int32),
        )
    )


def ship_at(x, y, theme, vx=0.0, active=True):
    return engine.entity(x, y, vx, 0.0, rx=1.0, ry=0.6, type=SHIP, image_type=SHIP, image_theme=theme, active=active)


def bullet_at(x, y, vy=0.0, active=True):
    return engine.entity(
        x,
        y,
        0.0,
        vy,
        rx=0.25,
        ry=0.25,
        type=PLAYER_BULLET,
        image_type=PLAYER_BULLET,
        collides_with_entities=True,
        expire_time=BULLET_EXPIRE,
        active=active,
    )


@pytest.fixture(scope='module')
def env():
    return PlunderEnv()


# --------------------------------------------------------------------------- #
# interface and the fleet
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)
    assert env.observation_size == 64 * 64 * 3


def test_reset_builds_a_sea_with_a_legend(env):
    state = jit_reset(jax.random.PRNGKey(0))
    board = state.env_state
    pool = board.entities
    active = np.asarray(pool.active)

    assert np.all(np.asarray(board.grid) == SPACE)
    types = np.asarray(pool.type)[active]
    assert types.tolist().count(TARGET_BACKGROUND) == 1
    legend = np.asarray(pool.image_theme)[active & (np.asarray(pool.type) == SHIP)]
    assert int(board.image_permutation[0]) in legend.tolist()


def test_the_target_is_one_of_the_six_ship_designs(env):
    state = jit_reset(jax.random.PRNGKey(0))
    board = state.env_state

    assert len(np.unique(np.asarray(board.image_permutation))) == NUM_SHIP_TYPES
    assert int(np.asarray(board.target_bools).sum()) == 1
    assert bool(board.target_bools[board.image_permutation[0]])


def test_the_player_is_not_the_target(env):
    for seed in range(6):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        theme = int(board.agent.image_theme)
        assert not bool(board.target_bools[theme]), 'the player starts as a friendly ship'


def test_panels_sit_in_the_middle_band(env):
    for seed in range(6):
        board = jit_reset(jax.random.PRNGKey(seed)).env_state
        pool = board.entities
        panels = np.asarray(pool.active) & (np.asarray(pool.type) == PANEL)
        ys = np.asarray(pool.y)[panels]
        assert np.all(ys >= 0.25 * WORLD_DIM - 1e-3)
        assert np.all(ys <= 0.5 * WORLD_DIM + 1e-3)


def test_five_lanes_with_alternating_speeds(env):
    board = jit_reset(jax.random.PRNGKey(0)).env_state

    assert board.lane_vels.shape == (NUM_LANES,)
    assert np.all(np.asarray(board.lane_vels) >= 0.15)
    assert np.all(np.asarray(board.lane_vels) <= 0.25)


def test_level_seed_pins_the_fleet(env):
    first = PlunderEnv(level_seed=5).reset(jax.random.PRNGKey(0)).env_state
    second = PlunderEnv(level_seed=5).reset(jax.random.PRNGKey(77)).env_state

    assert np.array_equal(np.asarray(first.image_permutation), np.asarray(second.image_permutation))


# --------------------------------------------------------------------------- #
# steering and firing
# --------------------------------------------------------------------------- #
def test_the_ship_only_moves_horizontally(env):
    state = jit_reset(jax.random.PRNGKey(0))
    start_y = float(state.env_state.agent.y)

    for action in (UP, DOWN):
        state = jit_step(state, jnp.asarray(action))
        assert float(state.env_state.agent.y) == start_y


def test_steering_right_accelerates(env):
    state = jit_reset(jax.random.PRNGKey(0))
    start = float(state.env_state.agent.x)

    for _ in range(8):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.x) > start


def test_the_ship_cannot_pass_the_legend(env):
    state = jit_reset(jax.random.PRNGKey(0))

    for _ in range(40):
        state = jit_step(state, jnp.asarray(LEFT))

    assert float(state.env_state.agent.x) >= float(state.env_state.min_agent_x) - 1e-5


def test_firing_spawns_a_bullet_on_a_cooldown(env):
    state = jit_reset(jax.random.PRNGKey(0))

    def bullets(state):
        return int((state.env_state.entities.active & (state.env_state.entities.type == PLAYER_BULLET)).sum())

    for step_index in range(3):
        state = jit_step(state, jnp.asarray(FIRE_ACTION))
        if step_index < 2:
            assert bullets(state) == 0, 'the cooldown blocks the first two steps'

    assert bullets(state) == 1
    pool = state.env_state.entities
    index = int(np.argmax(np.asarray(pool.type) == PLAYER_BULLET))
    assert float(pool.vy[index]) > 0, 'cannonballs travel upwards'
    assert int(pool.expire_time[index]) == BULLET_EXPIRE
    assert bool(pool.collides_with_entities[index])


def test_holding_fire_keeps_to_the_cooldown(env):
    state = jit_reset(jax.random.PRNGKey(0))

    for _ in range(12):
        state = jit_step(state, jnp.asarray(FIRE_ACTION))

    bullets = int((state.env_state.entities.active & (state.env_state.entities.type == PLAYER_BULLET)).sum())
    assert 3 <= bullets <= 4, 'one shot every three steps'


def test_firing_costs_juice(env):
    state = jit_reset(jax.random.PRNGKey(0))
    before = float(state.env_state.juice_left)

    for _ in range(3):
        state = jit_step(state, jnp.asarray(FIRE_ACTION))

    assert float(state.env_state.juice_left) < before


def test_bullets_expire(env):
    bullet = bullet_at(10.0, 10.0, vy=0.0)

    for _ in range(BULLET_EXPIRE + 1):
        bullet = engine.entity_step(bullet)

    assert not bool(bullet.active)


@pytest.mark.parametrize('action', [10, 11, 12, 13, 14])
def test_only_action_nine_fires(env, action):
    state = jit_reset(jax.random.PRNGKey(0))

    for _ in range(4):
        state = jit_step(state, jnp.asarray(action))

    bullets = int((state.env_state.entities.active & (state.env_state.entities.type == PLAYER_BULLET)).sum())
    assert bullets == 0


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_hitting_the_target_scores_and_refills(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    target_theme = int(np.flatnonzero(np.asarray(base.target_bools))[0])
    state = state_with(
        env,
        agent=base.agent._replace(x=jnp.asarray(10.0, jnp.float32), y=jnp.asarray(6.0, jnp.float32)),
        entities=[ship_at(10.0, 6.5, target_theme), bullet_at(10.0, 6.5)],
        juice=0.5,
    )

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == POSITIVE_REWARD
    assert int(stepped.env_state.targets_hit) == 1
    assert float(stepped.env_state.juice_left) > 0.5
    assert not bool(stepped.env_state.entities.active.any())


def test_hitting_a_friendly_ship_drains_juice(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    friendly = 0 if not bool(base.target_bools[0]) else 1
    state = state_with(
        env,
        agent=base.agent._replace(x=jnp.asarray(10.0, jnp.float32), y=jnp.asarray(6.0, jnp.float32)),
        entities=[ship_at(10.0, 6.5, friendly), bullet_at(10.0, 6.5)],
        juice=0.5,
    )

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == 0.0
    assert float(stepped.env_state.juice_left) < 0.5


def test_a_bullet_stops_on_a_panel(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    panel = engine.entity(10.0, 6.5, rx=1.2, ry=0.5, type=PANEL, image_type=PANEL, active=True)
    state = state_with(
        env,
        agent=base.agent._replace(x=jnp.asarray(10.0, jnp.float32), y=jnp.asarray(6.0, jnp.float32)),
        entities=[panel, bullet_at(10.0, 6.5)],
        juice=0.5,
    )

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert not bool(stepped.env_state.entities.active.any()), 'both the bullet and the crate go'
    assert float(stepped.reward) == 0.0


def test_empty_juice_ends_the_episode(env):
    state = state_with(env, juice=0.001)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])


def test_the_quota_pays_the_bonus(env):
    base = jit_reset_state(jax.random.PRNGKey(0))
    target_theme = int(np.flatnonzero(np.asarray(base.target_bools))[0])
    state = state_with(
        env,
        agent=base.agent._replace(x=jnp.asarray(10.0, jnp.float32), y=jnp.asarray(6.0, jnp.float32)),
        entities=[ship_at(10.0, 6.5, target_theme), bullet_at(10.0, 6.5)],
        juice=0.5,
        targets_hit=TARGET_QUOTA - 1,
    )

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == POSITIVE_REWARD + COMPLETION_BONUS
    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])


def test_timeout_truncates():
    env = PlunderEnv(timeout=5)
    state = state_with(env, steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])
    assert int(stepped.env_state.steps) == 0


def test_timeout_matches_the_original():
    assert TIMEOUT == 4000


# --------------------------------------------------------------------------- #
# traffic, observations, jit
# --------------------------------------------------------------------------- #
def test_ships_keep_arriving_and_stay_in_the_pool(env):
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        return carry, carry.env_state.entities.active.sum()

    _, alive = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((300,), NOOP)))(state)

    assert int(alive.max()) > 0
    assert int(alive.max()) < MAX_ENTITIES


def test_ships_face_the_way_they_drive(env):
    """``is_reflected = !moves_right``: the sprite mirror follows the lane direction."""
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        pool = carry.env_state.entities
        return carry, (pool.vx, pool.is_reflected, pool.active & (pool.type == SHIP))

    _, (vxs, reflected, ships) = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((120,), NOOP)))(state)
    vxs = np.asarray(vxs)[np.asarray(ships)]
    reflected = np.asarray(reflected)[np.asarray(ships)]

    moving = np.abs(vxs) > 1e-6
    assert moving.any()
    assert np.array_equal(reflected[moving], vxs[moving] < 0)


def test_observation_shows_the_sea_and_the_fleet(env):
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
