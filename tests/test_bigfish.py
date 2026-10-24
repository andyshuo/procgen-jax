"""ProcGen's ``bigfish``: swimming, eating, growing and dying.

The rules come from ``procgen/src/games/bigfish.cpp``: eat anything smaller than
you (+1 and you grow), die on contact with anything bigger, and finish after 30
fish (+10).  There is no terrain and no gravity; the world is a 20x20 arena.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.bigfish import (
    COMPLETION_BONUS,
    FISH,
    FISH_QUOTA,
    MAX_FISH,
    POSITIVE_REWARD,
    START_R,
    TIMEOUT,
    WATER,
    WORLD_DIM,
    BigFishEnv,
    empty_fish_pool,
)
from procgen_jax.objects import SPACE

# Actions: vx = a // 3 - 1, vy = a % 3 - 1.
UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7

JIT_ENV = BigFishEnv()
jit_step = jax.jit(JIT_ENV.step)

# Reset runs the level generator (and, for leaper, a 300 step warm-up);
# jitting it keeps the suite out of eager scans.
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)


def make_pool(entities) -> engine.Entity:
    pool = empty_fish_pool()
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, fish=(), agent=None, steps=0, fish_eaten=0):
    """Wrap a hand-written arena into the env's :class:`State`."""
    base = jit_reset_state(jax.random.PRNGKey(0))
    return env.state_from(
        base._replace(
            agent=base.agent if agent is None else agent,
            fish=make_pool(fish),
            steps=jnp.asarray(steps, jnp.int32),
            fish_eaten=jnp.asarray(fish_eaten, jnp.int32),
        )
    )


def fish_at(x, y, r=0.25, vx=0.0, theme=0):
    return engine.entity(x, y, vx, 0.0, rx=r, ry=r, type=FISH, image_type=FISH, image_theme=theme, active=True)


@pytest.fixture(scope='module')
def env():
    return BigFishEnv()


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #
def test_env_exposes_sizes(env):
    assert env.action_size == 15
    assert env.observation_shape == (64, 64, 3)
    assert env.observation_size == 64 * 64 * 3
    assert env.unwrapped is env


def test_reset_returns_a_state_with_an_observation(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert state.obs.shape == (64, 64, 3)
    assert state.obs.dtype == jnp.uint8
    assert np.isclose(float(state.env_state.agent.rx), START_R)
    assert np.isclose(float(state.env_state.agent.y), 1 + START_R)
    assert int(state.env_state.fish_eaten) == 0
    assert not bool(state.env_state.fish.active.any()), 'no fish at the start'


def test_the_grid_is_empty_and_the_world_is_twenty_tiles(env):
    state = jit_reset(jax.random.PRNGKey(0))

    assert state.env_state.grid.shape == (WORLD_DIM, WORLD_DIM)
    assert np.all(np.asarray(state.env_state.grid) == SPACE)


def test_agent_starts_inside_the_world(env):
    for seed in range(8):
        x = float(jit_reset(jax.random.PRNGKey(seed)).env_state.agent.x)
        assert 0.4 <= x <= WORLD_DIM - 0.4


def test_level_seed_pins_the_start(env):
    first = BigFishEnv(level_seed=7).reset(jax.random.PRNGKey(0))
    second = BigFishEnv(level_seed=7).reset(jax.random.PRNGKey(123))

    assert float(first.env_state.agent.x) == float(second.env_state.agent.x)


def test_different_keys_start_somewhere_else(env):
    xs = {round(float(jit_reset(jax.random.PRNGKey(seed)).env_state.agent.x), 3) for seed in range(8)}

    assert len(xs) > 1


# --------------------------------------------------------------------------- #
# swimming
# --------------------------------------------------------------------------- #
def test_pressing_up_moves_the_agent_up(env):
    """``vy = -1`` for action 3, and the arena has no gravity."""
    state = jit_step(jit_reset(jax.random.PRNGKey(0)), jnp.asarray(UP))

    assert float(state.env_state.agent.vy) < 0.0
    assert float(state.env_state.agent.y) < 1 + START_R


def test_pressing_right_moves_the_agent_right(env):
    state = jit_step(jit_reset(jax.random.PRNGKey(0)), jnp.asarray(RIGHT))

    assert float(state.env_state.agent.vx) > 0.0
    assert float(state.env_state.agent.x) > 0.0


def test_velocity_decays_when_no_action_is_given(env):
    state = jit_step(jit_reset(jax.random.PRNGKey(0)), jnp.asarray(RIGHT))
    fast = float(state.env_state.agent.vx)

    state = jit_step(state, jnp.asarray(NOOP))

    assert 0.0 < float(state.env_state.agent.vx) < fast, 'mixrate decays the velocity'


def test_the_agent_cannot_leave_the_world(env):
    """``out_of_bounds_object`` is ``INVALID_OBJ`` here, which blocks like a wall."""
    state = jit_reset(jax.random.PRNGKey(0))
    for _ in range(40):
        state = jit_step(state, jnp.asarray(UP))

    assert float(state.env_state.agent.y) >= float(state.env_state.agent.ry) - 1e-3
    assert not bool(state.done)


@pytest.mark.parametrize('action', [9, 10, 11, 12, 13, 14])
def test_action_slots_past_the_grid_are_noops(env, action):
    state = jit_reset(jax.random.PRNGKey(0))

    stepped = jit_step(state, jnp.asarray(action))

    assert float(stepped.env_state.agent.vx) == 0.0
    assert float(stepped.env_state.agent.vy) == 0.0


# --------------------------------------------------------------------------- #
# eating and dying
# --------------------------------------------------------------------------- #
def test_eating_a_smaller_fish_rewards_and_grows(env):
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(env, fish=[fish_at(float(agent.x), float(agent.y), r=0.25)], agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == POSITIVE_REWARD
    assert int(stepped.env_state.fish_eaten) == 1
    assert float(stepped.env_state.agent.rx) > float(agent.rx)
    assert not bool(stepped.env_state.fish.active.any()), 'the fish is eaten'
    assert not bool(stepped.done)


def test_touching_a_bigger_fish_ends_the_episode(env):
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(env, fish=[fish_at(float(agent.x), float(agent.y), r=1.0)], agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])
    assert float(stepped.reward) == 0.0


def test_a_fish_of_equal_size_is_eaten(env):
    """The original compares ``obj->rx > agent->rx``, so ties go to the agent."""
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(env, fish=[fish_at(float(agent.x), float(agent.y), r=START_R)], agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == POSITIVE_REWARD
    assert not bool(stepped.done)


def test_the_quota_pays_the_bonus_and_ends_the_episode(env):
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(
        env,
        fish=[fish_at(float(agent.x), float(agent.y), r=0.25)],
        agent=agent,
        fish_eaten=FISH_QUOTA - 1,
    )

    stepped = jit_step(state, jnp.asarray(NOOP))

    # The returned state is already a fresh episode, so the evidence is the reward.
    assert float(stepped.reward) == POSITIVE_REWARD + COMPLETION_BONUS
    assert bool(stepped.done)
    assert bool(stepped.info['terminated'])
    assert int(stepped.env_state.fish_eaten) == 0


def test_distant_fish_is_untouched(env):
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(env, fish=[fish_at(float(agent.x) + 5.0, float(agent.y), r=0.25)], agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert float(stepped.reward) == 0.0
    assert bool(stepped.env_state.fish.active.any())


# --------------------------------------------------------------------------- #
# spawning
# --------------------------------------------------------------------------- #
def test_fish_spawn_and_swim_across_the_arena(env):
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        return carry, carry.env_state.fish.active.sum()

    _, alive = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((300,), NOOP)))(state)

    assert int(alive.max()) > 0, 'fish should keep arriving'
    assert int(alive.max()) <= MAX_FISH


def test_spawned_fish_are_flat_and_inside_their_lane(env):
    """``match_aspect_ratio`` squashes the fish, and they enter from the edges."""
    state = jit_reset(jax.random.PRNGKey(0))

    def body(carry, action):
        carry = jit_step(carry, action)
        return carry, carry.env_state.fish

    _, fish = jax.jit(lambda s: jax.lax.scan(body, s, jnp.full((200,), NOOP)))(state)
    active = np.asarray(fish.active)
    rx = np.asarray(fish.rx)
    ry = np.asarray(fish.ry)

    assert active.any()
    assert np.all(ry[active] <= rx[active] + 1e-6), 'fish sprites are wider than tall'


# --------------------------------------------------------------------------- #
# termination
# --------------------------------------------------------------------------- #
def test_timeout_truncates():
    env = BigFishEnv(timeout=5)
    state = state_with(env, steps=4)

    stepped = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert bool(stepped.info['truncated'])
    assert not bool(stepped.info['terminated'])
    assert int(stepped.env_state.steps) == 0  # auto-reset


def test_auto_reset_clears_the_arena(env):
    agent = jit_reset_state(jax.random.PRNGKey(0)).agent
    state = state_with(env, fish=[fish_at(float(agent.x), float(agent.y), r=1.0)], agent=agent)

    stepped = jit_step(state, jnp.asarray(NOOP))

    assert bool(stepped.done)
    assert int(stepped.env_state.fish_eaten) == 0
    assert np.isclose(float(stepped.env_state.agent.rx), START_R)


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
def test_observation_shows_the_whole_world(env):
    """``center_agent`` is off, so the corners of the window are the arena."""
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)
    water = np.asarray(WATER, np.uint8)

    for row, col in ((0, 0), (0, 63), (63, 0), (63, 63)):
        assert np.array_equal(obs[row, col], water), (row, col)


def test_observation_draws_the_player(env):
    """The camera is fixed on the world, so the player is where its body is."""
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)
    water = np.asarray(WATER, np.uint8)
    unit = 64 / WORLD_DIM
    agent = state.env_state.agent
    row = int((WORLD_DIM - float(agent.y)) * unit)
    col = int(float(agent.x) * unit)

    assert not np.array_equal(obs[row, col], water), f'no sprite at ({row}, {col})'


# --------------------------------------------------------------------------- #
# jit / vmap / scan
# --------------------------------------------------------------------------- #
def test_step_is_jittable(env):
    state = jit_reset(jax.random.PRNGKey(0))

    stepped = jax.jit(env.step)(state, jnp.asarray(RIGHT))

    assert stepped.obs.shape == state.obs.shape
    assert jnp.isfinite(stepped.reward)
    assert stepped.done.dtype == jnp.bool_


def test_whole_episode_runs_under_scan(env):
    state = jit_reset(jax.random.PRNGKey(0))
    actions = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 15)

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, carry.reward

    final_state, rewards = jax.jit(lambda: jax.lax.scan(body, state, actions))()

    assert rewards.shape == (64,)
    assert jnp.isfinite(rewards).all()
    assert final_state.env_state.grid.shape == (WORLD_DIM, WORLD_DIM)


def test_vmap_reset_and_step(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = jax.vmap(env.reset)(keys)
    assert states.obs.shape == (4, 64, 64, 3)

    stepped = jax.vmap(env.step)(states, jnp.array([RIGHT, LEFT, UP, NOOP]))

    assert stepped.obs.shape == (4, 64, 64, 3)
    assert stepped.reward.shape == (4,)
    assert stepped.done.shape == (4,)


def test_vmap_batch_has_distinct_starts(env):
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))

    xs = {round(float(x), 3) for x in states.env_state.agent.x}
    assert len(xs) > 1


def test_timeout_matches_the_original():
    assert TIMEOUT == 6000, 'bigfish.cpp sets timeout = 6000'
