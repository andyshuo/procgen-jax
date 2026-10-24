"""A pure-JAX port of ProcGen's ``plunder``, from ``procgen/src/games/plunder.cpp``.

A top-down naval duel: your ship slides left and right along the bottom, firing
cannonballs upward with action 9.  One of the six ship designs is the current
target -- hits on it score +1 and refill the "juice" meter, hits on anything else
drain it.  Juice also drains steadily, so missing costs you the episode; twenty
target hits pays +10 and ends it.

Deviations: sprites are not rotated (the original turns ships by +/-90 degrees and
the target legend sideways), the on-screen juice and progress bars are not drawn,
and hit explosions are not spawned.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``plunder.cpp`` object ids.
PLAYER_BULLET = 1
TARGET_LEGEND = 2
TARGET_BACKGROUND = 3
PANEL = 6
SHIP = 7

COMPLETION_BONUS = 10.0
POSITIVE_REWARD = 1.0

WORLD_DIM = 20
TIMEOUT = 4000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.85
NUM_LANES = 5
NUM_SHIP_TYPES = 6
NUM_CURRENT_SHIP_TYPES = 2
TARGET_QUOTA = 20
SPAWN_PROB = 0.06
R_SCALE = 1.0
LEGEND_R = 2.0
KEY_SCALE = 1.5
PANEL_WIDTH = 1.2
#: ``juice_left -= 0.0015f`` per step, ``+0.1`` per target hit, ``-0.1`` otherwise.
JUICE_DECAY = 0.0015
JUICE_PER_TARGET = 0.1
#: Firing costs ``0.02`` and is gated by ``cur_time - last_fire_time >= 3``.
FIRE_COST = 0.02
FIRE_COOLDOWN = 3
#: Bullets carry ``expire_time = 50``; ships can fire one every third step at most.
BULLET_EXPIRE = 50
BULLET_RADIUS = 0.25
#: Ships, bullets, panels and the legend; steady state is well under 64.
MAX_ENTITIES = 64
MAX_SUB_STEPS = 6
#: ``reposition`` retries until it finds a free spot, up to this many times.
REPOSITION_ATTEMPTS = 100
#: Action 9 is "fire": ``special_action = action - 8``.
FIRE_ACTION = 9

SHIP_SPRITES = ['ship_1', 'ship_2', 'ship_3', 'ship_4', 'ship_5', 'ship_6']

TILE_SPRITES = jnp.full((64,), sprites.row('_transparent'), jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[PLAYER_BULLET, 0]
    .set(sprites.row('misc/plunder/cannonBall'))
    .at[PANEL, 0]
    .set(sprites.row('misc/plunder/panel_wood'))
    .at[TARGET_BACKGROUND, 0]
    .set(sprites.row('misc/plunder/target_red2'))
)
for _theme, _name in enumerate(SHIP_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[SHIP, _theme].set(sprites.row(f'misc/plunder/{_name}'))
SHIP_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/plunder/{name}')] for name in SHIP_SPRITES])


class PlunderState(NamedTuple):
    """Game state: the sea, the fleet, and the target bookkeeping."""

    grid: jax.Array  # (20, 20) int32, all SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,)
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937
    juice_left: jax.Array
    targets_hit: jax.Array
    lane_directions: jax.Array  # (NUM_LANES,) bool
    lane_vels: jax.Array  # (NUM_LANES,) float32
    target_bools: jax.Array  # (NUM_SHIP_TYPES,) bool, indexed by image_theme
    image_permutation: jax.Array  # (NUM_SHIP_TYPES,) int32
    last_fire_time: jax.Array
    r_scale: jax.Array
    legend_r: jax.Array
    min_agent_x: jax.Array


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class PlunderEnv(Env):
    """ProcGen's ``plunder`` as a functional, jit-friendly environment."""

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

        def reset(env_state: PlunderState) -> PlunderState:
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
    def _rand_pos(self, state, radius, low, high, active):
        """``rand_pos``: uniform in ``[low+radius, high-radius]``, or the midpoint."""
        draws = (high - low) > 2 * radius
        state, draw = jax.lax.cond(
            draws & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        value = jnp.where(draws, (high - low - 2 * radius) * draw + radius + low, (high + low) / 2)
        return state, value

    def _reposition(self, state, pool, agent, rx, ry, low_x, high_x, low_y, high_y, check_agent=True):
        """``reposition``: retry until the body is clear, consuming two draws each time."""
        start = (state, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32), jnp.zeros((), jnp.bool_))

        def attempt(index, carry):
            state, x, y, done = carry
            state, nx = self._rand_pos(state, rx, low_x, high_x, ~done)
            state, ny = self._rand_pos(state, ry, low_y, high_y, ~done)
            candidate = engine.entity(nx, ny, rx=rx, ry=ry, active=True)
            blocked = (engine.collides(candidate, pool) & pool.active).any()
            if check_agent:
                blocked = blocked | engine.collides(candidate, agent)
            fresh = ~done & ~blocked
            return (
                state,
                jnp.where(fresh, nx, x),
                jnp.where(fresh, ny, y),
                done | fresh,
            )

        state, x, y, _ = jax.lax.fori_loop(0, REPOSITION_ATTEMPTS, attempt, start)
        return state, x, y

    def reset_env_state(self, key: jax.Array) -> PlunderState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        # BasicAbstractGame::game_reset draws.
        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, start_x = rng.rand01(state)
        state, start_y = rng.rand01(state)

        # ``choose_n`` over all six ship types: one draw per pick, with the element
        # removed after each pick, so the modulus shrinks every time.
        remaining = jnp.arange(NUM_SHIP_TYPES, dtype=jnp.int32)
        count = jnp.asarray(NUM_SHIP_TYPES, jnp.int32)
        order = jnp.zeros((NUM_SHIP_TYPES,), jnp.int32)
        positions = jnp.arange(NUM_SHIP_TYPES)
        for pick_index in range(NUM_SHIP_TYPES):
            state, pick = rng.randn(state, count)
            order = order.at[pick_index].set(remaining[pick])
            shifted = jnp.concatenate([remaining[1:], jnp.zeros((1,), jnp.int32)])
            remaining = jnp.where(positions >= pick, shifted, remaining)
            count = count - 1
        image_permutation = order
        target_bools = jnp.zeros((NUM_SHIP_TYPES,), bool).at[order[0]].set(True)

        lane_directions = jnp.zeros((NUM_LANES,), bool)
        lane_vels = jnp.zeros((NUM_LANES,), jnp.float32)
        for lane in range(NUM_LANES):
            state, direction_draw = rng.rand01(state)
            state, speed_draw = rng.rand01(state)
            lane_directions = lane_directions.at[lane].set(direction_draw < 0.5)
            lane_vels = lane_vels.at[lane].set(0.15 + 0.1 * speed_draw)

        grid = jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32)
        pool = empty_pool()
        agent = engine.entity(
            start_x * (WORLD_DIM - 2 * 0.4) + 0.4,
            start_y * (WORLD_DIM - 2 * 0.4) + 0.4,
            rx=0.4,
            ry=0.4,
            type=engine.PLAYER,
            image_type=SHIP,
            render_z=1,
            smart_step=True,
        )
        next_slot = jnp.zeros((), jnp.int32)

        def write(pool, next_slot, ent):
            slot = next_slot % MAX_ENTITIES
            pool = jax.tree.map(lambda buf, value: buf.at[slot].set(value), pool, ent)
            return pool, next_slot + 1

        # Crates drift in the middle band.
        state, num_panels = rng.randn(state, 4)
        panels = jnp.arange(4) < num_panels
        for index in range(4):
            spawn = panels[index]
            state, x, y = jax.lax.cond(
                spawn,
                lambda s, pool=pool: self._reposition(
                    s, pool, agent, PANEL_WIDTH, 0.5, 0.0, WORLD_DIM, 0.25 * WORLD_DIM, 0.5 * WORLD_DIM
                ),
                lambda s: (s, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)),
                state,
            )
            panel = engine.entity(x, y, rx=PANEL_WIDTH, ry=0.5, type=PANEL, image_type=PANEL, active=spawn)
            pool, next_slot = write(pool, next_slot, panel)

        # The legend: a backdrop disc with the target ship on top.
        legend_r = jnp.asarray(LEGEND_R, jnp.float32)
        pool, next_slot = write(
            pool,
            next_slot,
            engine.entity(
                legend_r,
                legend_r,
                rx=legend_r,
                ry=legend_r,
                type=TARGET_BACKGROUND,
                image_type=TARGET_BACKGROUND,
                active=True,
            ),
        )
        legend_theme = image_permutation[0]
        pool, next_slot = write(
            pool,
            next_slot,
            engine.entity(
                legend_r,
                legend_r,
                rx=R_SCALE * KEY_SCALE,
                ry=R_SCALE * KEY_SCALE / SHIP_ASPECTS[legend_theme],
                type=SHIP,
                image_type=SHIP,
                image_theme=legend_theme,
                active=True,
            ),
        )

        # The player takes the non-target ship type.
        state, theme_draw = rng.randn(state, NUM_CURRENT_SHIP_TYPES // 2)
        player_theme = image_permutation[NUM_CURRENT_SHIP_TYPES // 2 + theme_draw]
        agent = agent._replace(
            image_theme=player_theme,
            ry=R_SCALE / SHIP_ASPECTS[player_theme],
        )

        state, x, y = self._reposition(
            state, pool, agent, R_SCALE, R_SCALE, 0.0, WORLD_DIM, 0.0, WORLD_DIM, check_agent=False
        )
        agent = agent._replace(x=x, y=1 + R_SCALE / SHIP_ASPECTS[player_theme])
        min_agent_x = 2 * LEGEND_R + R_SCALE
        agent = agent._replace(x=jnp.maximum(agent.x, min_agent_x))

        return PlunderState(
            grid=grid,
            agent=agent,
            entities=pool,
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
            juice_left=jnp.ones((), jnp.float32),
            targets_hit=jnp.zeros((), jnp.int32),
            lane_directions=lane_directions,
            lane_vels=lane_vels,
            target_bools=target_bools,
            image_permutation=image_permutation,
            last_fire_time=jnp.zeros((), jnp.int32),
            r_scale=jnp.asarray(R_SCALE, jnp.float32),
            legend_r=legend_r,
            min_agent_x=jnp.asarray(min_agent_x, jnp.float32),
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: PlunderState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        special_action = jnp.where(action >= 9, action - 8, 0)

        # ---- update_agent_velocity: horizontal only ------------------------ #
        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        agent = agent._replace(vx=0.9 * vx, vy=0.0)

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(-1),
            is_blocked=lambda tile, is_horizontal: tile == -1,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=jnp.zeros((), jnp.int32),
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # ---- step_entities and expiry -------------------------------------- #
        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- handle_collision: bullets only -------------------------------- #
        def collide_bullets(index, carry):
            pool, reward, hits, juice = carry
            is_bullet = pool.collides_with_entities[index] & pool.active[index]
            bullet = jax.tree.map(lambda field: field[index], pool)
            hit = engine.collides(bullet, pool) & pool.active & is_bullet
            hit = hit.at[index].set(False)

            ship = hit & (pool.type == SHIP)
            panel = hit & (pool.type == PANEL)
            target = (pool.type == SHIP) & env_state.target_bools[pool.image_theme]
            killed = (ship | panel).any()
            scored = (ship & target).any()
            missed = (ship & ~target).any()

            pool = pool._replace(
                active=jnp.where(ship | panel, False, pool.active).at[index].set(pool.active[index] & ~killed)
            )
            return (
                pool,
                reward + jnp.where(scored, POSITIVE_REWARD, 0.0),
                hits + jnp.where(scored, 1, 0),
                juice + jnp.where(scored, JUICE_PER_TARGET, 0.0) - jnp.where(missed, JUICE_PER_TARGET, 0.0),
            )

        pool, reward, hits, juice = jax.lax.fori_loop(
            0,
            MAX_ENTITIES,
            collide_bullets,
            (pool, jnp.zeros((), jnp.float32), env_state.targets_hit, env_state.juice_left - JUICE_DECAY),
        )

        # ---- spawn: one draw per step, two more when a ship appears -------- #
        state, spawn_draw = rng.rand01(env_state.rng)
        spawn = spawn_draw < SPAWN_PROB

        def do_spawn(state):
            state, lane = rng.randn(state, NUM_LANES)
            state, theme_pick = rng.randn(state, NUM_CURRENT_SHIP_TYPES)
            return state, (lane, theme_pick)

        def no_spawn(state):
            zero = jnp.zeros((), jnp.int32)
            return state, (zero, zero)

        state, (lane, theme_pick) = jax.lax.cond(spawn, do_spawn, no_spawn, state)
        moves_right = env_state.lane_directions[lane]
        ent_y = (lane * 0.11 + 0.4) * (WORLD_DIM / 2 - R_SCALE) + WORLD_DIM / 2
        ent_vx = env_state.lane_vels[lane] * jnp.where(moves_right, 1.0, -1.0)
        theme = env_state.image_permutation[theme_pick]
        new_ship = engine.entity(
            jnp.where(moves_right, -R_SCALE, WORLD_DIM + R_SCALE),
            ent_y,
            ent_vx,
            0.0,
            rx=R_SCALE,
            ry=R_SCALE / SHIP_ASPECTS[theme],
            type=SHIP,
            image_type=SHIP,
            image_theme=theme,
            is_reflected=~moves_right,
            active=spawn,
        )
        free = pool.active == 0
        slot = jnp.argmax(free)
        overlapping = (engine.collides(new_ship, pool) & pool.active).any() | engine.collides(new_ship, agent)
        write = new_ship.active & free.any() & ~overlapping
        new_ship = new_ship._replace(active=write)
        pool = jax.lax.cond(
            write,
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, new_ship),
            lambda p: p,
            pool,
        )

        # ---- fire ---------------------------------------------------------- #
        can_fire = (special_action == 1) & ((env_state.steps + 1 - env_state.last_fire_time) >= FIRE_COOLDOWN)
        bullet = engine.entity(
            agent.x,
            agent.y,
            0.0,
            1.0,
            rx=BULLET_RADIUS,
            ry=BULLET_RADIUS,
            type=PLAYER_BULLET,
            image_type=PLAYER_BULLET,
            collides_with_entities=True,
            expire_time=BULLET_EXPIRE,
            active=can_fire,
        )
        free = pool.active == 0
        bullet_slot = jnp.argmax(free)
        pool = jax.lax.cond(
            can_fire & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[bullet_slot].set(value), p, bullet),
            lambda p: p,
            pool,
        )
        juice = juice - jnp.where(can_fire, FIRE_COST, 0.0)
        juice = jnp.minimum(juice, 1.0)
        last_fire_time = jnp.where(can_fire, env_state.steps + 1, env_state.last_fire_time)

        # ---- juice, quota, the legend barrier ------------------------------ #
        agent = agent._replace(x=jnp.maximum(agent.x, env_state.min_agent_x))
        killed = juice <= 0
        quota = hits >= TARGET_QUOTA
        reward = reward + jnp.where(quota, COMPLETION_BONUS, 0.0)
        terminated = killed | quota | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)

        stepped = PlunderState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            steps=env_state.steps + 1,
            key=key,
            rng=state,
            juice_left=juice,
            targets_hit=hits,
            lane_directions=env_state.lane_directions,
            lane_vels=env_state.lane_vels,
            target_bools=env_state.target_bools,
            image_permutation=env_state.image_permutation,
            last_fire_time=last_fire_time,
            r_scale=env_state.r_scale,
            legend_r=env_state.legend_r,
            min_agent_x=env_state.min_agent_x,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: PlunderState,
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

    def observe(self, env_state: PlunderState) -> jax.Array:
        """The whole sea: ``center_agent`` is off, so all 20 tiles are visible."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        centre = jnp.asarray([WORLD_DIM / 2, WORLD_DIM / 2])
        return render_textured(
            env_state.grid,
            centre,
            bodies,
            TILE_SPRITES,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            WORLD_DIM,
            jnp.asarray(SPACE),
        )
