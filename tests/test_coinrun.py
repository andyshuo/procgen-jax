"""ProcGen's ``coinrun``: level generation, platformer physics and termination."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine
from procgen_jax.games.coinrun import (
    AGENT_RX,
    AGENT_RY,
    CRATE,
    ENEMY1,
    ENEMY2,
    ENTITY_SPRITES,
    GOAL,
    GOAL_REWARD,
    LAVA_MID,
    LAVA_TOP,
    MAX_ENTITIES,
    MAX_JUMP,
    PLAYER_JUMP,
    PLAYER_RIGHT1,
    PLAYER_RIGHT2,
    SAW,
    SAW2,
    START_X,
    START_Y,
    TILE_SPRITES,
    WALL_MID,
    WALL_TOP,
    WORLD_DIM,
    CoinRunEnv,
    CoinRunState,
    empty_entity_pool,
    generate_level,
)
from procgen_jax.render import SKY_COLOUR

# Actions: vx = a // 3 - 1, vy = a % 3 - 1.
UP, LEFT, DOWN, NOOP, RIGHT = 3, 1, 5, 4, 7
#: Slot 5 (the "down" row) is the jump button in ProcGen's coinrun.
JUMP = DOWN


def make_pool(entities) -> engine.Entity:
    pool = empty_entity_pool(MAX_ENTITIES)
    for index, ent in enumerate(entities):
        pool = jax.tree.map(lambda buf, val, i=index: buf.at[i].set(val), pool, ent)
    return pool


def state_with(env, grid, agent=None, enemies=(), statics=(), steps=0, last_agent_y=None, is_on_crate=False):
    """Wrap a hand-written grid into the env's :class:`State`."""
    if agent is None:
        agent = engine.entity(
            START_X, START_Y, rx=AGENT_RX, ry=AGENT_RY, type=engine.PLAYER, render_z=1, smart_step=True
        )
    return env.state_from(
        CoinRunState(
            grid=jnp.asarray(grid, jnp.int32),
            agent=agent,
            enemies=make_pool(enemies),
            statics=make_pool(statics),
            num_enemies=jnp.asarray(len(enemies), jnp.int32),
            num_statics=jnp.asarray(len(statics), jnp.int32),
            steps=jnp.asarray(steps, jnp.int32),
            key=jax.random.PRNGKey(0),
            last_agent_y=jnp.asarray(agent.y if last_agent_y is None else last_agent_y, jnp.float32),
            has_support=jnp.asarray(True),
            is_on_crate=jnp.asarray(is_on_crate),
            facing_right=jnp.asarray(True),
            wall_theme=jnp.asarray(0),
        )
    )


def flat_grid():
    """The level's bottom row is the floor, as ``init_floor_and_walls`` sets it."""
    return jnp.zeros((WORLD_DIM, WORLD_DIM), jnp.int32).at[0, :].set(WALL_TOP)


level = jax.jit(generate_level)


@pytest.fixture(scope='module')
def env():
    return CoinRunEnv()


# Eager ``env.step`` runs hundreds of tiny ops per step, so the physics tests
# drive a jitted step. It lives at module scope so the trace is shared.
JIT_ENV = CoinRunEnv()
jit_step = jax.jit(JIT_ENV.step)

# Reset runs the level generator (and, for leaper, a 300 step warm-up);
# jitting it keeps the suite out of eager scans.
jit_reset = jax.jit(JIT_ENV.reset)
jit_reset_state = jax.jit(JIT_ENV.reset_env_state)


# --------------------------------------------------------------------------- #
# level generation
# --------------------------------------------------------------------------- #
def test_generate_level_shapes():
    grid, enemies, statics, num_enemies, num_statics, _, _ = level(0)

    assert grid.shape == (WORLD_DIM, WORLD_DIM)
    assert grid.dtype == jnp.int32
    assert enemies.x.shape == (MAX_ENTITIES,)
    assert statics.x.shape == (MAX_ENTITIES,)
    assert 0 <= int(num_enemies) < MAX_ENTITIES
    assert 0 <= int(num_statics) < MAX_ENTITIES


def test_level_has_a_floor_and_exactly_one_coin():
    grid = np.asarray(level(0)[0])

    assert np.all(grid[0] != 0), 'the bottom row must be solid floor'
    assert int((grid == GOAL).sum()) == 1


def test_coin_sits_on_a_plinth():
    grid = np.asarray(level(3)[0])
    coin_y, coin_x = np.argwhere(grid == GOAL)[0]

    assert grid[coin_y - 1, coin_x] in (WALL_MID, WALL_TOP)


@pytest.mark.parametrize('seed', range(12))
def test_every_level_keeps_the_agent_start_clear(seed):
    """The agent starts standing on the floor, not inside geometry."""
    grid = np.asarray(level(seed)[0])

    assert grid[0, 1] == WALL_TOP
    assert grid[1, 1] == 0


def test_generation_is_deterministic():
    first = np.asarray(level(42)[0])
    second = np.asarray(level(42)[0])

    assert np.array_equal(first, second)


def test_generation_differs_across_keys():
    first = np.asarray(level(1)[0])
    second = np.asarray(level(2)[0])

    assert not np.array_equal(first, second)


def test_entity_pools_never_overflow():
    """The original appends to a vector; here the pools are fixed size."""
    for seed in range(32):
        _, _, _, num_enemies, num_statics, _, _ = level(seed)
        assert int(num_enemies) < MAX_ENTITIES
        assert int(num_statics) < MAX_ENTITIES


def test_generated_entities_sit_inside_the_world():
    for seed in range(8):
        _, enemies, statics, num_enemies, num_statics, _, _ = level(seed)
        for pool, count in ((enemies, num_enemies), (statics, num_statics)):
            active = np.asarray(pool.active[:count])
            xs = np.asarray(pool.x[:count])
            ys = np.asarray(pool.y[:count])
            assert active.all()
            assert np.all((xs > 0) & (xs < WORLD_DIM))
            assert np.all((ys > 0) & (ys < WORLD_DIM))


def test_a_greedy_policy_can_reach_the_coin(env):
    """The structural tests cannot show that a level can be finished at all.

    Pressing right and jumping every fifth step is enough on every seed tried;
    this pins that down so a generator change that walls the coin off fails here.
    """
    seeds, steps = 3, 260
    actions = jnp.full((steps,), RIGHT, jnp.int32).at[::5].set(JUMP)

    def run(keys):
        states = jax.vmap(env.reset)(keys)

        def body(carry, action):
            carry = jax.vmap(env.step)(carry, jnp.broadcast_to(action, (seeds,)))
            return carry, carry.reward

        return jax.lax.scan(body, states, actions)[1]

    rewards = jax.jit(run)(jax.random.split(jax.random.PRNGKey(0), seeds))
    reached = int((rewards.sum(axis=0) >= GOAL_REWARD).sum())

    assert reached == seeds


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
    assert not bool(state.done)
    assert float(state.reward) == 0.0
    assert np.isclose(float(state.env_state.agent.x), START_X)
    assert np.isclose(float(state.env_state.agent.y), START_Y)


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def test_standing_still_keeps_the_agent_put(env):
    state = state_with(env, flat_grid())

    for _ in range(5):
        state = jit_step(state, jnp.asarray(NOOP))

    assert np.isclose(float(state.env_state.agent.y), START_Y)
    assert float(state.env_state.agent.vx) == 0.0


def test_jump_reaches_the_discrete_peak(env):
    """``vy`` is 1.5, then drops by gravity, so the peak is 1.5+1.3+...+0.1 = 6.4."""
    state = state_with(env, flat_grid())

    state = jit_step(state, jnp.asarray(JUMP))
    peak = float(state.env_state.agent.y)
    for _ in range(20):
        state = jit_step(state, jnp.asarray(NOOP))
        peak = max(peak, float(state.env_state.agent.y))

    assert np.isclose(peak, START_Y + 6.4, atol=1e-3)
    assert np.isclose(float(state.env_state.agent.y), START_Y, atol=1e-4), 'should land again'


def test_a_second_jump_needs_support(env):
    """Pressing jump in mid-air does nothing: ``set_action_xy`` zeroes it."""
    state = state_with(env, flat_grid())
    state = jit_step(state, jnp.asarray(JUMP))
    rising = float(state.env_state.agent.vy)

    # Hold jump while airborne: the next step must lose speed to gravity, not reset it.
    state = jit_step(state, jnp.asarray(JUMP))

    assert float(state.env_state.agent.vy) < rising


def test_walking_accelerates_towards_max_speed(env):
    state = state_with(env, flat_grid())

    for _ in range(12):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.vx) > 0.4
    assert float(state.env_state.agent.x) > 3.0


def test_walls_block_the_agent(env):
    grid = flat_grid().at[1, 2].set(WALL_MID)
    state = state_with(env, grid)

    for _ in range(8):
        state = jit_step(state, jnp.asarray(RIGHT))

    assert float(state.env_state.agent.x) < 2.0


def test_crates_are_landable_from_above(env):
    """``is_blocked_ents`` makes crates a platform, but only while falling."""
    crate = engine.entity(1.5, 2.5, type=CRATE)
    agent = engine.entity(1.5, 4.5787, rx=AGENT_RX, ry=AGENT_RY, type=engine.PLAYER, render_z=1, smart_step=True)
    state = state_with(env, flat_grid(), agent=agent, statics=[crate], last_agent_y=4.5787)

    for _ in range(6):
        state = jit_step(state, jnp.asarray(NOOP))

    # The crate's top is y = 3.0, so the agent rests at 3.0 + ry.
    assert np.isclose(float(state.env_state.agent.y), 3.0 + AGENT_RY, atol=1e-4)


# --------------------------------------------------------------------------- #
# termination
# --------------------------------------------------------------------------- #
def test_the_coin_rewards_and_terminates(env):
    grid = flat_grid().at[1, 2].set(GOAL)
    state = state_with(env, grid)

    for _ in range(6):
        state = jit_step(state, jnp.asarray(RIGHT))
        if bool(state.done):
            break

    assert bool(state.info['terminated'])
    assert float(state.reward) == GOAL_REWARD
    assert not bool(state.info['truncated'])


def test_lava_kills(env):
    grid = flat_grid().at[1, 2].set(LAVA_TOP).at[2, 2].set(LAVA_MID)
    state = state_with(env, grid)

    for _ in range(6):
        state = jit_step(state, jnp.asarray(RIGHT))
        if bool(state.done):
            break

    assert bool(state.info['terminated'])
    assert float(state.reward) == 0.0


def test_touching_an_enemy_kills(env):
    enemy = engine.entity(2.5, 1.5, vx=0.0, type=5, render_z=1, smart_step=True)
    state = state_with(env, flat_grid(), enemies=[enemy])

    for _ in range(6):
        state = jit_step(state, jnp.asarray(RIGHT))
        if bool(state.done):
            break

    assert bool(state.info['terminated'])


def test_touching_a_saw_kills(env):
    saw = engine.entity(2.5, 1.5, type=SAW)
    state = state_with(env, flat_grid(), statics=[saw])

    for _ in range(6):
        state = jit_step(state, jnp.asarray(RIGHT))
        if bool(state.done):
            break

    assert bool(state.info['terminated'])


def test_goal_transition_auto_resets(env):
    grid = flat_grid().at[1, 2].set(GOAL)
    state = state_with(env, grid)

    for _ in range(6):
        state = jit_step(state, jnp.asarray(RIGHT))
        if bool(state.done):
            break

    assert bool(state.done), 'gymnax semantics: done describes the transition'
    assert np.isclose(float(state.env_state.agent.x), START_X), 'returned state is a fresh episode'
    assert int(state.env_state.steps) == 0


def test_timeout_truncates():
    env = CoinRunEnv(timeout=5)  # the shared jit_step uses the default timeout
    state = state_with(env, flat_grid(), steps=4)

    state = jax.jit(env.step)(state, jnp.asarray(NOOP))

    assert bool(state.done)
    assert bool(state.info['truncated'])
    assert not bool(state.info['terminated'])
    assert int(state.env_state.steps) == 0  # auto-reset


@pytest.mark.parametrize('action', [9, 10, 11, 12, 13, 14])
def test_action_slots_past_the_grid_do_not_move(env, action):
    state = state_with(env, flat_grid())

    stepped = jit_step(state, jnp.asarray(action))

    assert np.isclose(float(stepped.env_state.agent.x), START_X)
    assert float(stepped.reward) == 0.0


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
def test_observation_draws_the_player_at_the_centre(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    # The player sprite is centred in its own window, so nothing else can be there.
    assert not np.array_equal(obs[32, 32], np.asarray(SKY_COLOUR, np.uint8))


def test_observation_pads_outside_the_world_with_walls(env):
    """The agent starts at x = 1.5, so the left edge of the window is outside."""
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert not np.array_equal(obs[32, 0], np.asarray(SKY_COLOUR, np.uint8))


def test_observation_shows_the_floor_below_the_agent(env):
    state = jit_reset(jax.random.PRNGKey(0))
    obs = np.asarray(state.obs)

    assert not np.array_equal(obs[37, 32], np.asarray(SKY_COLOUR, np.uint8))


def test_sprite_tables_cover_every_object_the_game_draws():
    """A missing atlas entry would silently render as nothing."""
    transparent = 0
    for tile in (GOAL, WALL_MID, WALL_TOP, LAVA_MID, LAVA_TOP):
        assert all(int(TILE_SPRITES[theme, tile]) != transparent for theme in range(TILE_SPRITES.shape[0]))
    assert int(TILE_SPRITES[0, 19]) == transparent  # ENEMY_BARRIER is invisible

    for image_type, themes in (
        (engine.PLAYER, 5),
        (PLAYER_JUMP, 5),
        (PLAYER_RIGHT1, 5),
        (PLAYER_RIGHT2, 5),
        (ENEMY1, 9),
        (ENEMY2, 9),
        (SAW, 1),
        (SAW2, 1),
        (CRATE, 4),
    ):
        for theme in range(themes):
            assert int(ENTITY_SPRITES[image_type, theme]) != transparent, (image_type, theme)


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
    actions = jax.random.randint(jax.random.PRNGKey(1), (32,), 0, 15)

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, carry.reward

    final_state, rewards = jax.jit(lambda: jax.lax.scan(body, state, actions))()

    assert rewards.shape == (32,)
    assert jnp.isfinite(rewards).all()
    assert final_state.env_state.grid.shape == (WORLD_DIM, WORLD_DIM)


def test_vmap_reset_and_step(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    states = jax.vmap(env.reset)(keys)
    assert states.obs.shape == (4, 64, 64, 3)

    actions = jnp.array([RIGHT, LEFT, JUMP, NOOP])
    stepped = jax.vmap(env.step)(states, actions)

    assert stepped.obs.shape == (4, 64, 64, 3)
    assert stepped.reward.shape == (4,)
    assert stepped.done.shape == (4,)


def test_vmap_batch_has_distinct_levels(env):
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))

    grids = np.asarray(states.env_state.grid)
    assert len({grid.tobytes() for grid in grids}) > 1


def test_max_jump_matches_the_engine_bound():
    """``MAX_JUMP`` is what keeps ``int(4*|v|)`` inside the sub-step budget."""
    speed = float(jnp.sqrt(0.5**2 + MAX_JUMP**2))
    assert int(4 * speed) <= 6
