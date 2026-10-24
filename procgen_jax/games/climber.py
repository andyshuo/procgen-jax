"""A pure-JAX port of ProcGen's ``climber``, from ``procgen/src/games/climber.cpp``.

A vertical platformer: platforms climb a tall shaft, each one may hold a crystal and
a swimming enemy patrols a few tiles either side of where it spawned.  Collecting
every crystal ends the level for ten.

Deviations: no enemy barrier tiles exist in the level (the original reserves the id
but never places one), and the sprite does not rotate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``climber.cpp`` object ids.
COIN = 1
ENEMY = 5
ENEMY1 = 6
ENEMY2 = 7
WALL_MID = 15
WALL_TOP = 16
PLAYER_JUMP = 9
PLAYER_RIGHT1 = 12
PLAYER_RIGHT2 = 13
#: ProcGen's ``SPACE``, which the level keeps in its grid.
SPACE = 100

COIN_REWARD = 1.0
COMPLETION_BONUS = 10.0
WORLD_WIDTH = 20
WORLD_HEIGHT = 64
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.5
GRAVITY = 0.2
MAX_JUMP = 1.5
AIR_CONTROL = 0.15
TERMINAL_VELOCITY = -2.0
PATROL_RANGE = 4
NUM_WALL_THEMES = 4
AGENT_RADIUS = 0.5
COIN_RADIUS = 0.3
ENEMY_RADIUS = 0.5
MAX_PLATFORM_LEN = 12
MAX_PLATFORMS = 10
MAX_ENTITIES = 32
MAX_SUB_STEPS = 6
#: ``choose_delta_y``: ``max_jump^2 / (2 * gravity)`` is 5.625, so 5 - 3 + 1 draws.
DELTA_Y_RANGE = 3
MIN_DELTA_Y = 3

PLAYER_FRAMES = ['stand', 'walk1', 'walk2', 'walk4']
PLAYER_COLOURS = ['Blue', 'Green', 'Grey', 'Red']

TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32)
    .at[WALL_MID]
    .set(sprites.row('platformer/tileBlue_08'))
    .at[WALL_TOP]
    .set(sprites.row('platformer/tileBlue_05'))
)
WALL_ROWS = jnp.asarray(
    [sprites.row(f'platformer/{name}') for name in ('tileBlue_08', 'tileGreen_08', 'tileYellow_09', 'tileBrown_09')],
    jnp.int32,
)
WALL_TOP_ROWS = jnp.asarray(
    [sprites.row(f'platformer/{name}') for name in ('tileBlue_05', 'tileGreen_05', 'tileYellow_06', 'tileBrown_06')],
    jnp.int32,
)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[COIN, 0]
    .set(sprites.row('platformer/yellowCrystal'))
    .at[ENEMY1, 0]
    .set(sprites.row('platformer/enemySwimming_1'))
    .at[ENEMY2, 0]
    .set(sprites.row('platformer/enemySwimming_2'))
)
for _theme, _colour in enumerate(PLAYER_COLOURS):
    ENTITY_SPRITES = ENTITY_SPRITES.at[engine.PLAYER, _theme].set(sprites.row(f'platformer/player{_colour}_stand'))
    ENTITY_SPRITES = ENTITY_SPRITES.at[PLAYER_JUMP, _theme].set(sprites.row(f'platformer/player{_colour}_walk4'))
    ENTITY_SPRITES = ENTITY_SPRITES.at[PLAYER_RIGHT1, _theme].set(sprites.row(f'platformer/player{_colour}_walk1'))
    ENTITY_SPRITES = ENTITY_SPRITES.at[PLAYER_RIGHT2, _theme].set(sprites.row(f'platformer/player{_colour}_walk2'))
#: The original paints a platformer background; this is a flat stand-in.
SKY_COLOUR = (48, 64, 96)


class ClimberState(NamedTuple):
    """Game state: the shaft, the climbing player, the crystals and the swimmers."""

    grid: jax.Array  # (WORLD_HEIGHT * WORLD_WIDTH,) WALL_MID, WALL_TOP or SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) crystals and enemies
    patrol_origin: jax.Array  # (MAX_ENTITIES,) where each enemy started
    wall_theme: jax.Array
    has_support: jax.Array
    facing_right: jax.Array
    coin_quota: jax.Array
    coins_collected: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class ClimberEnv(Env):
    """ProcGen's ``climber`` as a functional, jit-friendly environment."""

    def __init__(self, timeout: int = TIMEOUT, level_seed: int | None = None):
        self.timeout = timeout
        self.level_seed = level_seed

    # -- Env interface ----------------------------------------------------- #
    @property
    def action_size(self) -> int:
        return NUM_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (64, 64, 3)

    def reset(self, key: jax.Array) -> State:
        return self.state_from(self.reset_env_state(key))

    def step(self, state: State, action: jax.Array) -> State:
        env_state, reward, terminated = self.step_env_state(state.env_state, action)
        truncated = jnp.logical_and(env_state.steps >= self.timeout, jnp.logical_not(terminated))
        done = jnp.logical_or(terminated, truncated)

        def reset(env_state: ClimberState) -> ClimberState:
            reset_key, _ = jax.random.split(env_state.key)
            return self.reset_env_state(reset_key)

        return self.state_from(
            auto_reset(env_state, done, reset),
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_obs=state.obs,
        )

    # -- level generation --------------------------------------------------- #
    def _place(self, flat, x, y, dx, dy, value):
        xs = jnp.arange(WORLD_WIDTH)
        ys = jnp.arange(WORLD_HEIGHT)
        inside = (xs[None, :] >= x) & (xs[None, :] < x + dx) & (ys[:, None] >= y) & (ys[:, None] < y + dy)
        return jnp.where(inside.reshape(-1), value, flat)

    def reset_env_state(self, key: jax.Array) -> ClimberState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        size = WORLD_HEIGHT * WORLD_WIDTH
        flat = jnp.full((size,), SPACE, jnp.int32)
        # ``init_floor_and_walls``: a floor of wall tops, sides and a ceiling
        flat = self._place(flat, 0, 0, WORLD_WIDTH, 1, WALL_TOP)
        flat = self._place(flat, 0, 0, 1, WORLD_HEIGHT, WALL_MID)
        flat = self._place(flat, WORLD_WIDTH - 1, 0, 1, WORLD_HEIGHT, WALL_MID)
        flat = self._place(flat, 0, WORLD_HEIGHT - 1, WORLD_WIDTH, 1, WALL_MID)

        state, difficulty = rng.randn(state, 3)
        min_platforms = difficulty * difficulty + 1
        max_platforms = (difficulty + 1) * (difficulty + 1) + 1
        state, platform_draw = rng.randn(state, max_platforms - min_platforms + 1)
        num_platforms = platform_draw + min_platforms

        state, start_draw = rng.randn(state, WORLD_WIDTH - 4)
        curr_x = start_draw + 2
        curr_y = jnp.zeros((), jnp.int32)

        pool = empty_pool()
        patrol = jnp.zeros((MAX_ENTITIES,), jnp.float32)
        slot = jnp.zeros((), jnp.int32)
        coin_quota = jnp.zeros((), jnp.int32)

        for platform in range(MAX_PLATFORMS):
            live = platform < num_platforms

            state, delta_draw = jax.lax.cond(
                live,
                lambda st: rng.randn(st, DELTA_Y_RANGE),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            delta_y = delta_draw + MIN_DELTA_Y

            can_spawn = live & (curr_x >= 3) & (curr_x <= WORLD_WIDTH - 3)
            state, enemy_draw = jax.lax.cond(
                can_spawn, lambda st: rng.rand01(st), lambda st: (st, jnp.ones((), jnp.float32)), state
            )
            spawn_enemy = can_spawn & (enemy_draw < 0.5)
            state, enemy_speed = jax.lax.cond(
                spawn_enemy, lambda st: rng.randn(st, 2), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            state, enemy_lift = jax.lax.cond(
                spawn_enemy, lambda st: rng.randn(st, 2), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            enemy = engine.entity(
                curr_x.astype(jnp.float32) + 0.5,
                (curr_y + enemy_lift + 2).astype(jnp.float32) + 0.5,
                0.15 * (enemy_speed * 2 - 1),
                0.0,
                rx=ENEMY_RADIUS,
                ry=ENEMY_RADIUS,
                type=ENEMY,
                image_type=ENEMY1,
                smart_step=True,
                active=spawn_enemy,
            )
            position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=position, on=spawn_enemy: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                enemy,
            )
            patrol = patrol.at[position].set(jnp.where(spawn_enemy, curr_x.astype(jnp.float32) + 0.5, patrol[position]))
            slot = slot + spawn_enemy

            curr_y = curr_y + delta_y
            state, length_draw = jax.lax.cond(
                live, lambda st: rng.randn(st, 10), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            platform_len = length_draw + 2
            state, dir_draw = jax.lax.cond(
                live, lambda st: rng.randn(st, 2), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            direction = dir_draw * 2 - 1
            direction = jnp.where(curr_x < 3, 1, jnp.where(curr_x > WORLD_WIDTH - 3, -1, direction))

            # lay the platform's tiles, stopping at the walls
            candidates = jnp.full((MAX_PLATFORM_LEN,), -1, jnp.int32)
            count = jnp.zeros((), jnp.int32)
            stopped = jnp.zeros((), bool)
            for index in range(MAX_PLATFORM_LEN):
                nx = curr_x + (index + 1) * direction
                ok = live & ~stopped & (index < platform_len) & (nx > 0) & (nx < WORLD_WIDTH - 1)
                flat = jnp.where(ok, flat.at[(curr_y * WORLD_WIDTH + nx).reshape(())].set(WALL_TOP), flat)
                candidates = candidates.at[index].set(jnp.where(ok, nx, candidates[index]))
                count = count + ok
                stopped = stopped | (~ok & (index < platform_len))

            # a crystal on some platforms, and the anchor for the next one
            state, coin_draw = jax.lax.cond(
                live, lambda st: rng.rand01(st), lambda st: (st, jnp.ones((), jnp.float32)), state
            )
            with_coin = live & ((coin_draw < 0.5) | (platform == num_platforms - 1))
            candidates_count = jnp.maximum(count, 1)
            state, coin_pick = jax.lax.cond(
                with_coin,
                lambda st, n=candidates_count: rng.randn(st, n),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            state, next_pick = jax.lax.cond(
                live,
                lambda st, n=candidates_count: rng.randn(st, n),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            coin_x = candidates[jnp.clip(coin_pick, 0, MAX_PLATFORM_LEN - 1)]
            coin = engine.entity(
                coin_x.astype(jnp.float32) + 0.5,
                (curr_y + 1).astype(jnp.float32) + 0.5,
                rx=COIN_RADIUS,
                ry=COIN_RADIUS,
                type=COIN,
                image_type=COIN,
                active=with_coin,
            )
            position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=position, on=with_coin: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                coin,
            )
            patrol = patrol.at[position].set(jnp.where(with_coin, 0.0, patrol[position]))
            slot = slot + with_coin
            coin_quota = coin_quota + with_coin

            curr_x = candidates[jnp.clip(next_pick, 0, MAX_PLATFORM_LEN - 1)]
            curr_x = jnp.where(live, curr_x, curr_x)

        state, theme = rng.randn(state, len(PLAYER_COLOURS))
        state, wall_theme = rng.randn(state, NUM_WALL_THEMES)

        agent = engine.entity(
            1 + AGENT_RADIUS,
            1 + AGENT_RADIUS,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            image_theme=theme,
            render_z=1,
            smart_step=True,
        )
        return ClimberState(
            grid=flat,
            agent=agent,
            entities=pool,
            patrol_origin=patrol,
            wall_theme=wall_theme,
            has_support=jnp.asarray(False),
            facing_right=jnp.asarray(True),
            coin_quota=coin_quota,
            coins_collected=jnp.zeros((), jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: ClimberState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1
        view = env_state.grid.reshape(WORLD_HEIGHT, WORLD_WIDTH)
        agent = env_state.agent

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = jnp.maximum(move_action % 3 - 1, 0)

        # ---- set_action_xy ----------------------------------------------------- #
        below = []
        for offset in (AGENT_RADIUS - 0.01, -(AGENT_RADIUS - 0.01)):
            cell_x = jnp.clip(jnp.floor(agent.x + offset).astype(jnp.int32), 0, WORLD_WIDTH - 1)
            cell_y = jnp.clip(jnp.floor(agent.y - (AGENT_RADIUS + 0.01)).astype(jnp.int32), 0, WORLD_HEIGHT - 1)
            below.append(view[cell_y, cell_x])
        support_tiles = jnp.stack(below)
        has_support = ((support_tiles == WALL_MID) | (support_tiles == WALL_TOP)).any()
        action_vy = jnp.where(has_support & (action_vy == 1), 1, 0)
        facing_right = jnp.where(action_vx > 0, True, jnp.where(action_vx < 0, False, env_state.facing_right))

        # ---- update_agent_velocity -------------------------------------------- #
        mixrate_x = jnp.where(has_support, MIXRATE, MIXRATE * AIR_CONTROL)
        vx = (1 - mixrate_x) * agent.vx + mixrate_x * MAX_SPEED * action_vx
        vy = jnp.where(action_vy > 0, MAX_JUMP, agent.vy)
        vy = jnp.where((~has_support) & (vy > TERMINAL_VELOCITY), vy - GRAVITY, vy)
        reflected = jnp.where(action_vx > 0, False, jnp.where(action_vx < 0, True, env_state.agent.is_reflected))
        agent = agent._replace(vx=vx, vy=vy, is_reflected=reflected)

        agent, _ = engine.step_object(
            agent,
            view,
            out_of_bounds=jnp.asarray(WALL_MID),
            is_blocked=lambda tile, is_horizontal: (tile == WALL_MID) | (tile == WALL_TOP),
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- the swimmers patrol their spawn point ---------------------------- #
        is_enemy = pool.active & (pool.type == ENEMY)
        origin = env_state.patrol_origin
        vx = jnp.where(is_enemy & (pool.x > origin + PATROL_RANGE), -jnp.abs(pool.vx), pool.vx)
        vx = jnp.where(is_enemy & (pool.x < origin - PATROL_RANGE), jnp.abs(vx), vx)
        pool = pool._replace(
            vx=vx,
            image_type=jnp.where(is_enemy, jnp.where(now // 5 % 2 == 0, ENEMY1, ENEMY2), pool.image_type),
            is_reflected=jnp.where(is_enemy, vx < 0, pool.is_reflected),
        )

        # ---- contact ----------------------------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        ate_coin = touching & (pool.type == COIN)
        hit_enemy = touching & (pool.type == ENEMY)
        pool = pool._replace(active=pool.active & ~ate_coin)
        coins_collected = env_state.coins_collected + ate_coin.sum()
        reward = COIN_REWARD * ate_coin.sum().astype(jnp.float32)

        finished = coins_collected == env_state.coin_quota
        reward = reward + jnp.where(finished, COMPLETION_BONUS, 0.0)
        terminated = hit_enemy.any() | finished

        image_type = jnp.where(
            ~has_support,
            PLAYER_JUMP,
            jnp.where(
                (jnp.abs(agent.vx) < 0.01) & (action_vx == 0),
                engine.PLAYER,
                jnp.where(now // 5 % 2 == 0, PLAYER_RIGHT1, PLAYER_RIGHT2),
            ),
        )
        stepped = ClimberState(
            grid=env_state.grid,
            agent=agent._replace(image_type=image_type),
            entities=pool,
            patrol_origin=env_state.patrol_origin,
            wall_theme=env_state.wall_theme,
            has_support=has_support,
            facing_right=facing_right,
            coin_quota=env_state.coin_quota,
            coins_collected=coins_collected,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: ClimberState,
        reward: jax.Array | float = 0.0,
        done: jax.Array | bool = False,
        terminated: jax.Array | bool = False,
        truncated: jax.Array | bool = False,
        final_obs: jax.Array | None = None,
    ) -> State:
        obs = self.observe(env_state)
        return State(
            env_state=env_state,
            obs=obs,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            done=jnp.asarray(done),
            info={
                'terminated': jnp.asarray(terminated),
                'truncated': jnp.asarray(truncated),
                'final_obs': obs if final_obs is None else final_obs,
            },
        )

    def observe(self, env_state: ClimberState) -> jax.Array:
        """``choose_center``: the shaft is ``main_width`` across and follows the climb."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        tile_rows = (
            TILE_ROWS.at[WALL_MID]
            .set(WALL_ROWS[env_state.wall_theme])
            .at[WALL_TOP]
            .set(WALL_TOP_ROWS[env_state.wall_theme])
        )
        centre = jnp.asarray(
            [
                WORLD_WIDTH / 2,
                env_state.agent.y + WORLD_WIDTH / 2 - 5 * AGENT_RADIUS,
            ]
        )
        return render_textured(
            env_state.grid.reshape(WORLD_HEIGHT, WORLD_WIDTH),
            centre,
            bodies,
            tile_rows,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            WORLD_WIDTH,
            jnp.asarray(SPACE),
            background=SKY_COLOUR,
        )
