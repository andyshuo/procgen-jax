"""A pure-JAX port of ProcGen's ``leaper``, from ``procgen/src/games/leaper.cpp``.

A frogger board: five-ish lanes of road traffic and water, the frog hops one tile
per press (velocity burst, then a fixed decay), must not be hit by a car, and must
ride the logs across the water -- standing still on open water drowns.  Crossing
the finish line while stationary pays +10.

The hop is the interesting part: pressing a direction sets the velocity to
``maxspeed`` and the same step immediately decays it, so the frog moves
0.4 + 0.3 + 0.2 + 0.1 = exactly one tile over ``NSTEP = 5`` steps.

Sprites are not rotated (the original rotates cars by PI and the frog by its
direction), so a car heading left is mirrored and the frog always faces up; the
finish line is stretched rather than tiled.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``leaper.cpp`` object ids.
LOG = 1
ROAD = 2
WATER = 3
CAR = 4
FINISH_LINE = 5
#: ``GOAL_REWARD = 10.0f``, ``timeout = 500``.
GOAL_REWARD = 10.0
TIMEOUT = 500
NUM_ACTIONS = 15
#: ``NSTEP = 5``: ``MAX_SPEED = 2 / (NSTEP - 1)``, ``VEL_DECAY = MAX_SPEED / NSTEP``.
NSTEP = 5
MAX_SPEED = 2 / (NSTEP - 1.0)
VEL_DECAY = MAX_SPEED / NSTEP
MONSTER_RADIUS = 0.25
LOG_RADIUS = 0.45
#: HardMode world; the observation shows all of it (``center_agent = false``).
WORLD_DIM = 15
#: ``difficulty <= 4`` and at most one extra lane.
MAX_LANES = 5
#: Steady state is about 50 bodies (logs sit ~2 tiles apart, cars ~6).
MAX_ENTITIES = 96
#: ``int(4 * |v|)`` with ``|v| <= maxspeed = 0.5``.
MAX_SUB_STEPS = 4
#: ``choose_extra_space()`` is ``randn(2)`` in hard mode.
MAX_EXTRA_SPACE = 2
#: ``get_adjusted_image_rect`` draws the frog 1.55 tiles tall.
FROG_SPRITE_HEIGHT = 1.55
#: ``main_width / std::min(min_car_speed, min_log_speed)`` = 15 / 0.05.
WARMUP_STEPS = 300

CAR_SPRITES = ['car_yellow_5', 'car_black_1', 'car_blue_2', 'car_green_3', 'car_red_4']
FROG_SPRITES = ['frog1', 'frog2', 'frog4', 'frog6', 'frog7']

#: Water and road tiles, indexed by tile id.
TILE_SPRITES = (
    jnp.full((64,), sprites.row('_transparent'), jnp.int32)
    .at[ROAD]
    .set(sprites.row('misc/leaper/roadTile6b'))
    .at[WATER]
    .set(sprites.row('misc/leaper/terrainTile6'))
)

ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[LOG, 0]
    .set(sprites.row('misc/leaper/elementWood044'))
    .at[FINISH_LINE, 0]
    .set(sprites.row('misc/leaper/finish2'))
)
for _theme, _name in enumerate(CAR_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[CAR, _theme].set(sprites.row(f'misc/leaper/{_name}'))
for _theme, _name in enumerate(FROG_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[engine.PLAYER, _theme].set(sprites.row(f'misc/leaper/{_name}'))


class LeaperState(NamedTuple):
    """Game state: the board, the frog, the traffic, and the lane speeds."""

    grid: jax.Array  # (15, 15) int32: SPACE / ROAD / WATER
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) cars and logs
    finish: engine.Entity
    road_speeds: jax.Array  # (MAX_LANES,) float32, y = bottom_road_y + lane
    water_speeds: jax.Array
    num_road_lanes: jax.Array
    num_water_lanes: jax.Array
    bottom_road_y: jax.Array
    bottom_water_y: jax.Array
    goal_y: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937
    next_slot: jax.Array


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def fill(grid: jax.Array, y, value) -> jax.Array:
    """``fill_elem(0, y, main_width, 1, value)``."""
    rows = jnp.arange(WORLD_DIM)[:, None]  # (row, 1) so this selects rows, not columns
    return jnp.where((rows >= y) & (rows < y + 1), jnp.asarray(value, jnp.int32), grid)


class LeaperEnv(Env):
    """ProcGen's ``leaper`` as a functional, jit-friendly environment."""

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

        def reset(env_state: LeaperState) -> LeaperState:
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
    def reset_env_state(self, key: jax.Array) -> LeaperState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        # BasicAbstractGame::game_reset draws, in order.
        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, agent_x = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        grid = jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32)

        # ---- road ---------------------------------------------------------- #
        state, extra = rng.randn(state, MAX_EXTRA_SPACE)
        bottom_road_y = extra + 1
        state, difficulty_draw = rng.randn(state, 5)
        difficulty = difficulty_draw
        state, lane_option = rng.randn(state, 4)
        num_road_lanes = difficulty + (lane_option == 2)

        road_speeds = jnp.zeros((MAX_LANES,), jnp.float32)
        for lane in range(MAX_LANES):
            lane_active = lane < num_road_lanes
            state, sign_draw = jax.lax.cond(
                lane_active,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.zeros((), jnp.float32)),
                state,
            )
            state, speed_draw = jax.lax.cond(
                lane_active,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.zeros((), jnp.float32)),
                state,
            )
            speed = jnp.where(sign_draw < 0.5, 1.0, -1.0) * (speed_draw * (0.2 - 0.05) + 0.05)
            road_speeds = road_speeds.at[lane].set(jnp.where(lane_active, speed, 0.0))
            grid = jnp.where(lane_active, fill(grid, bottom_road_y + lane, ROAD), grid)

        # ---- water --------------------------------------------------------- #
        state, extra = rng.randn(state, MAX_EXTRA_SPACE)
        bottom_water_y = bottom_road_y + num_road_lanes + extra + 1
        num_water_lanes = difficulty + (lane_option == 3)
        state, curr_sign_draw = rng.rand01(state)
        water_speeds = jnp.zeros((MAX_LANES,), jnp.float32)
        for lane in range(MAX_LANES):
            lane_active = lane < num_water_lanes
            state, speed_draw = jax.lax.cond(
                lane_active,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.zeros((), jnp.float32)),
                state,
            )
            # Lanes alternate direction, starting from rand_sign().
            direction = jnp.where(curr_sign_draw < 0.5, 1.0, -1.0) * jnp.where(lane % 2 == 0, 1.0, -1.0)
            speed = direction * (speed_draw * (0.1 - 0.05) + 0.05)
            water_speeds = water_speeds.at[lane].set(jnp.where(lane_active, speed, 0.0))
            grid = jnp.where(lane_active, fill(grid, bottom_water_y + lane, WATER), grid)

        goal_y = bottom_water_y + num_water_lanes + 1

        agent = engine.entity(
            agent_x * (WORLD_DIM - 2 * 0.4) + 0.4,
            0.375,
            rx=0.375,
            ry=0.375,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            image_theme=0,
            smart_step=True,
        )
        finish = engine.entity(
            WORLD_DIM / 2.0,
            goal_y - 0.5,
            rx=WORLD_DIM / 2.0,
            ry=0.5,
            type=FINISH_LINE,
            image_type=FINISH_LINE,
            active=True,
        )

        pool = empty_pool()
        next_slot = jnp.zeros((), jnp.int32)
        geometry = (bottom_road_y, num_road_lanes, bottom_water_y, num_water_lanes)

        # ---- warm the traffic up ------------------------------------------- #
        # The original runs ``main_width / min(min_car_speed, min_log_speed)``
        # spawn-and-step iterations before the agent is placed.
        def warmup(carry, _):
            pool, next_slot, state = carry
            pool, next_slot, state = self.spawn(pool, next_slot, state, road_speeds, water_speeds, geometry)
            pool = self.drift(pool)
            return (pool, next_slot, state), None

        (pool, next_slot, state), _ = jax.lax.scan(warmup, (pool, next_slot, state), None, length=WARMUP_STEPS)

        return LeaperState(
            grid=grid,
            agent=agent,
            entities=pool,
            finish=finish,
            road_speeds=road_speeds,
            water_speeds=water_speeds,
            num_road_lanes=num_road_lanes,
            num_water_lanes=num_water_lanes,
            bottom_road_y=bottom_road_y,
            bottom_water_y=bottom_water_y,
            goal_y=goal_y,
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
            next_slot=next_slot,
        )

    # -- traffic ------------------------------------------------------------ #
    def spawn(self, pool, next_slot, state, road_speeds, water_speeds, geometry):
        """``spawn_entities``: one draw per lane, plus a theme draw per car."""
        bottom_road_y, num_road_lanes, bottom_water_y, num_water_lanes = geometry
        new_entities = []

        for lane in range(MAX_LANES):
            active_lane = lane < num_road_lanes
            state, draw = jax.lax.cond(
                active_lane,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.ones((), jnp.float32)),
                state,
            )
            speed = road_speeds[lane]
            spawn = active_lane & (draw < jnp.abs(speed) / 6.0)
            state, theme = jax.lax.cond(
                spawn,
                lambda s: rng.randn(s, len(CAR_SPRITES)),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            car = engine.entity(
                jnp.where(speed > 0, -MONSTER_RADIUS, WORLD_DIM + MONSTER_RADIUS),
                bottom_road_y + lane + 0.5,
                speed,
                0.0,
                rx=2 * MONSTER_RADIUS,
                ry=MONSTER_RADIUS,
                type=CAR,
                image_type=CAR,
                image_theme=theme,
                is_reflected=speed < 0,
                active=spawn,
            )
            new_entities.append(car)

        for lane in range(MAX_LANES):
            active_lane = lane < num_water_lanes
            state, draw = jax.lax.cond(
                active_lane,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.ones((), jnp.float32)),
                state,
            )
            speed = water_speeds[lane]
            spawn = active_lane & (draw < jnp.abs(speed) / 2.0)
            log = engine.entity(
                jnp.where(speed > 0, -LOG_RADIUS, WORLD_DIM + LOG_RADIUS),
                bottom_water_y + lane + 0.5,
                speed,
                0.0,
                rx=LOG_RADIUS,
                type=LOG,
                image_type=LOG,
                active=spawn,
            )
            new_entities.append(log)

        # Write the spawned bodies into free slots, skipping any that would overlap
        # something already on the board (the original's ``has_any_collision``).
        count = len(new_entities)
        bodies = jax.tree.map(lambda *fields: jnp.stack(fields), *new_entities)
        for index in range(count):
            body = jax.tree.map(lambda field, i=index: field[i], bodies)
            free = pool.active == 0
            slot = jnp.argmax(free)
            has_free = free.any()
            overlapping = (engine.collides(body, pool) & pool.active).any()
            write = body.active & has_free & ~overlapping
            body = body._replace(active=write)
            pool = jax.tree.map(lambda buf, value, s=slot: buf.at[s].set(value), pool, body)
            next_slot = next_slot + write.astype(jnp.int32)
        return pool, next_slot, state

    def drift(self, pool) -> engine.Entity:
        """``step_entities`` for non-smart bodies, then erase the ones that left."""
        pool = pool._replace(x=jnp.where(pool.active, pool.x + pool.vx, pool.x))
        outside = engine.is_out_of_bounds(pool, WORLD_DIM, WORLD_DIM)
        return pool._replace(active=pool.active & ~outside)

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: LeaperState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        agent = env_state.agent
        # Animate the frog while it is mid-hop.
        frame = jnp.where(agent.image_theme >= 1, (agent.image_theme + 1) % NSTEP, agent.image_theme)
        agent = agent._replace(image_theme=frame)

        # ---- update_agent_velocity: a hop is a burst, then a fixed decay --- #
        idle = (agent.vx == 0) & (agent.vy == 0)
        hop_x = idle & (action_vx != 0)
        hop_y = idle & (action_vy != 0) & ~hop_x
        vx = jnp.where(hop_x, MAX_SPEED * action_vx, agent.vx)
        vy = jnp.where(hop_y, MAX_SPEED * action_vy, agent.vy)

        def decay(velocity):
            magnitude = jnp.maximum(jnp.abs(velocity) - VEL_DECAY, 0.0)
            return magnitude * jnp.sign(velocity)

        agent = agent._replace(
            vx=decay(vx),
            vy=decay(vy),
            image_theme=jnp.where(hop_x | hop_y, 1, frame),
        )

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(SPACE),
            is_blocked=lambda tile, is_horizontal: tile == engine.WALL_OBJ,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # ---- step_entities: traffic drifts --------------------------------- #
        pool = self.drift(env_state.entities)

        # ---- handle_agent_collision ---------------------------------------- #
        hit_car = (engine.collides(agent, pool) & pool.active & (pool.type == CAR)).any()
        at_finish = engine.collides(agent, env_state.finish)
        stopped = (agent.vx == 0) & (agent.vy == 0)
        crossed = at_finish & stopped
        reward = jnp.where(crossed, GOAL_REWARD, 0.0)

        # ---- logs carry the frog ------------------------------------------- #
        # ``has_collision(agent, log, margin=-agent.rx)``: only a real overlap counts.
        riding = engine.collides(agent, pool, -agent.rx) & pool.active & (pool.type == LOG)
        log_vx = jnp.max(pool.vx * riding, initial=0.0)
        agent = agent._replace(x=agent.x + jnp.where(riding.any(), log_vx, 0.0))

        # ---- water, edges, spawn ------------------------------------------- #
        tile = engine.tile_at(env_state.grid, agent.x, agent.y, jnp.asarray(SPACE))
        drowned = (tile == WATER) & ~riding.any() & stopped
        outside = engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)

        pool, next_slot, state = self.spawn(
            pool,
            env_state.next_slot,
            env_state.rng,
            env_state.road_speeds,
            env_state.water_speeds,
            (env_state.bottom_road_y, env_state.num_road_lanes, env_state.bottom_water_y, env_state.num_water_lanes),
        )

        terminated = hit_car | crossed | drowned | outside
        stepped = LeaperState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            finish=env_state.finish,
            road_speeds=env_state.road_speeds,
            water_speeds=env_state.water_speeds,
            num_road_lanes=env_state.num_road_lanes,
            num_water_lanes=env_state.num_water_lanes,
            bottom_road_y=env_state.bottom_road_y,
            bottom_water_y=env_state.bottom_water_y,
            goal_y=env_state.goal_y,
            steps=env_state.steps + 1,
            key=key,
            rng=state,
            next_slot=next_slot,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: LeaperState,
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

    def observe(self, env_state: LeaperState) -> jax.Array:
        """The whole board: ``center_agent`` is off, so all 15 tiles are visible."""
        bodies = jax.tree.map(
            lambda agent, pool, finish: jnp.concatenate([agent[None], pool, finish[None]]),
            env_state.agent,
            env_state.entities,
            env_state.finish,
        )
        is_frog = bodies.image_type == engine.PLAYER
        centre = jnp.asarray([WORLD_DIM / 2, WORLD_DIM / 2])
        return render_textured(
            env_state.grid,
            centre,
            bodies,
            TILE_SPRITES,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.where(is_frog, FROG_SPRITE_HEIGHT, 1.0),
            sprites.PIXELS,
            WORLD_DIM,
            jnp.asarray(SPACE),
        )
