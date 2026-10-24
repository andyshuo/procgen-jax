"""A pure-JAX port of ProcGen's ``starpilot``, from ``procgen/src/games/starpilot.cpp``.

A side-scrolling shooter: flyers, meteors, clouds and turrets drift in from the
edges and shoot back; every kill pays one point, and surviving to step 500 sends a
rocket across the screen that ends the level for ten more.

The level is a *spawn timeline*: ``add_spawners`` walks a clock up to
``SHOOTER_WIN_TIME``, drawing a group of enemies at every tick, so the port keeps a
fixed-size spawner table and spawns whatever shares the current step.

Deviations: explosions are not spawned (they are visual only) and the scrolling
space background is flat.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``starpilot.cpp`` object ids.
BULLET_PLAYER = 1
BULLET2 = 2
BULLET3 = 3
FLYER = 4
METEOR = 5
CLOUD = 6
TURRET = 7
FAST_FLYER = 8
FINISH_LINE = 9
NUM_BASIC_OBJECTS = 9

V_SCALE = 2.0 / 5.0
ENEMY_REWARD = 1.0
COMPLETION_BONUS = 10.0
SHOOTER_WIN_TIME = 500
NUM_SHIP_THEMES = 7

WORLD_DIM = 16
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.75
#: The player's gun fires on every step it is held.
PLAYER_BULLET_RADIUS = 0.4
#: Enemies, their bullets and the rocket; the timeline spends at most ~250 spawners.
MAX_ENTITIES = 160
MAX_SPAWNERS = 256
#: ``add_spawners`` advances its clock by at least ten, so it loops at most 50 times.
MAX_TIMELINE_STEPS = 64
MAX_SUB_STEPS = 6
#: ``init_hps`` for hard mode.
HP_VS = [1.0] * NUM_BASIC_OBJECTS
HP_HEALTH = [1.0] * NUM_BASIC_OBJECTS
HP_WEIGHT = [1.0] * NUM_BASIC_OBJECTS
HP_RADIUS = [0.5] * NUM_BASIC_OBJECTS

HP_VS[BULLET2] = 2.0
HP_VS[FAST_FLYER] = 1.5
HP_VS[BULLET_PLAYER] = 2.0
HP_VS[BULLET3] = 2.0
HP_HEALTH[TURRET] = 5.0
HP_HEALTH[FLYER] = 2.0
HP_HEALTH[FAST_FLYER] = 1.0
HP_HEALTH[METEOR] = 500.0
HP_HEALTH[CLOUD] = 0.0
HP_HEALTH[BULLET2] = 0.0
HP_HEALTH[BULLET3] = 0.0
HP_HEALTH[BULLET_PLAYER] = 0.0
HP_WEIGHT[FLYER] = 3.0
HP_WEIGHT[BULLET_PLAYER] = 0.0
HP_WEIGHT[BULLET2] = 0.0
HP_WEIGHT[BULLET3] = 0.0
HP_RADIUS[TURRET] = 2.0
HP_RADIUS[METEOR] = 2.0
HP_RADIUS[CLOUD] = 2.0
SLOW_V = 0.5
MAX_GROUP_SIZE = 5
MIN_DELTA_T = 10
MAX_DELTA_T = 30
SPAWN_RIGHT_THRESHOLD = 0.9
TOTAL_WEIGHT = sum(HP_WEIGHT[2:])
CAN_SPAWN_LEFT = True

VS = jnp.asarray(HP_VS, jnp.float32)
HEALTH = jnp.asarray(HP_HEALTH, jnp.float32)
WEIGHT = jnp.asarray(HP_WEIGHT, jnp.float32)
RADIUS = jnp.asarray(HP_RADIUS, jnp.float32)

FLYER_SPRITES = [f'spaceShips_{index:03d}' for index in range(1, 8)]
METEOR_SPRITES = [f'spaceMeteors_{index:03d}' for index in range(1, 5)] + [
    f'meteorGrey_big{index}' for index in range(1, 5)
]
CLOUD_SPRITES = [f'spaceEffect{index}' for index in range(1, 10)]
TURRET_SPRITES = ['spaceStation_018', 'spaceStation_019']
ROCKET_SPRITES = [f'spaceRockets_{index:03d}' for index in range(1, 5)]

TILE_SPRITES = jnp.full((64,), sprites.row('_transparent'), jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 10), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/starpilot/playerShip2_blue'))
    .at[BULLET_PLAYER, 0]
    .set(sprites.row('misc/starpilot/towerDefense_tile295'))
    .at[BULLET2, 0]
    .set(sprites.row('misc/starpilot/towerDefense_tile296'))
    .at[BULLET3, 0]
    .set(sprites.row('misc/starpilot/towerDefense_tile297'))
)
for _table, _type, _names in (
    (FLYER_SPRITES, FLYER, FLYER_SPRITES),
    (FLYER_SPRITES, FAST_FLYER, FLYER_SPRITES),
    (METEOR_SPRITES, METEOR, METEOR_SPRITES),
    (CLOUD_SPRITES, CLOUD, CLOUD_SPRITES),
    (TURRET_SPRITES, TURRET, TURRET_SPRITES),
    (ROCKET_SPRITES, FINISH_LINE, ROCKET_SPRITES),
):
    for _theme, _name in enumerate(_names):
        ENTITY_SPRITES = ENTITY_SPRITES.at[_type, _theme].set(sprites.row(f'misc/starpilot/{_name}'))

ASPECTS = {
    FLYER: jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/starpilot/{name}')] for name in FLYER_SPRITES]),
    METEOR: jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/starpilot/{name}')] for name in METEOR_SPRITES]),
    CLOUD: jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/starpilot/{name}')] for name in CLOUD_SPRITES]),
    TURRET: jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/starpilot/{name}')] for name in TURRET_SPRITES]),
}
#: The original paints space black.
SPACE_COLOUR = (0, 0, 0)


class StarPilotState(NamedTuple):
    """Game state: the ship, the enemies in play, and the unspawned timeline."""

    grid: jax.Array  # (WORLD_DIM, WORLD_DIM), all SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) enemies, bullets and the rocket
    spawners: engine.Entity  # (MAX_SPAWNERS,) timeline slots
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class StarPilotEnv(Env):
    """ProcGen's ``starpilot`` as a functional, jit-friendly environment."""

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

        def reset(env_state: StarPilotState) -> StarPilotState:
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
    def reset_env_state(self, key: jax.Array) -> StarPilotState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        spawners, state = self._build_timeline(state)

        # ``choose_random_theme(agent)``: one draw, the player ship has one theme.
        state, _ = rng.randn(state, 1)

        agent = engine.entity(
            WORLD_DIM / 2,
            WORLD_DIM / 2,
            rx=0.5,
            ry=0.5,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            smart_step=True,
        )
        return StarPilotState(
            grid=jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32),
            agent=agent,
            entities=empty_pool(MAX_ENTITIES),
            spawners=spawners,
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    def _rand_pos(self, state, radius, high, active):
        """``rand_pos(r, high)`` == ``rand_pos(r, 0, high)``: one draw unless the span is tight."""
        draws = high > 2 * radius
        state, draw = jax.lax.cond(
            draws & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        return state, jnp.where(draws, (high - 2 * radius) * draw + radius, high / 2)

    def _build_timeline(self, state):
        """``add_spawners``: walk the clock to ``SHOOTER_WIN_TIME`` filling a table."""
        pool = empty_pool(MAX_SPAWNERS)
        count = jnp.zeros((), jnp.int32)
        # ``t = 1 + randint(min_delta, max_delta)``
        state, t_draw = rng.randint(state, MIN_DELTA_T, MAX_DELTA_T)
        t = 1 + t_draw

        def round_body(_, carry):
            state, pool, count, t = carry
            active = t <= SHOOTER_WIN_TIME

            state, weight_draw = rng.rand01(state)
            start_weight = weight_draw * TOTAL_WEIGHT
            # Pick a type by walking the weights, as the C++ loop does.
            cumulative = jnp.cumsum(WEIGHT[2:])
            index = jnp.argmax(cumulative >= start_weight)
            kind = jnp.clip(index + 2, 2, NUM_BASIC_OBJECTS - 1)
            kind = jnp.where(start_weight <= 0, 2, kind)

            is_flyer = (kind == FLYER) | (kind == FAST_FLYER)
            state, group_draw = jax.lax.cond(
                is_flyer & active,
                lambda s: rng.randint(s, 0, MAX_GROUP_SIZE),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            group_size = jnp.where(is_flyer, group_draw + 1, 1)
            state, theme_draw = jax.lax.cond(
                is_flyer & active,
                lambda s: rng.randn(s, NUM_SHIP_THEMES),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            flyer_theme = theme_draw

            radius = RADIUS[kind]
            state, y_pos = self._rand_pos(state, radius, float(WORLD_DIM), active)

            def unit_body(unit, inner):
                state, pool, count = inner
                live = active & (unit < group_size)
                state, fire_draw = jax.lax.cond(
                    live, lambda s: rng.randint(s, 10, 100), lambda s: (s, jnp.zeros((), jnp.int32)), state
                )
                state, theta_draw = jax.lax.cond(
                    live, lambda s: rng.rand01(s), lambda s: (s, jnp.zeros((), jnp.float32)), state
                )
                state, zero_draw = jax.lax.cond(
                    live, lambda s: rng.randint(s, 0, 2), lambda s: (s, jnp.zeros((), jnp.int32)), state
                )
                theta = jnp.where(zero_draw == 1, 0.0, (theta_draw - 0.5) * (2 * jnp.pi / 4))
                v_scale = VS[kind]
                fire_time = fire_draw

                is_slow = (kind == METEOR) | (kind == CLOUD)
                state, turret_fire = jax.lax.cond(
                    live & (kind == TURRET),
                    lambda s: rng.randint(s, 20, 30),
                    lambda s: (s, jnp.zeros((), jnp.int32)),
                    state,
                )
                theta = jnp.where(is_slow | (kind == TURRET), 0.0, theta)
                v_scale = jnp.where(is_slow | (kind == TURRET), SLOW_V, v_scale)
                fire_time = jnp.where(kind == TURRET, turret_fire, fire_time)
                fire_time = jnp.where(is_slow, -1, fire_time)

                state, side_draw = jax.lax.cond(
                    live & is_flyer,
                    lambda s: rng.rand01(s),
                    lambda s: (s, jnp.zeros((), jnp.float32)),
                    state,
                )
                spawn_left = (side_draw > SPAWN_RIGHT_THRESHOLD) & CAN_SPAWN_LEFT & is_flyer
                state, theme_pick = jax.lax.cond(
                    live & (is_slow | (kind == TURRET)),
                    lambda s: rng.randn(s, jnp.where(kind == CLOUD, 9, jnp.where(kind == TURRET, 2, 8))),
                    lambda s: (s, jnp.zeros((), jnp.int32)),
                    state,
                )

                # The C++ writes the table entry even when the round is over.
                v_scale = v_scale * V_SCALE
                vx = -jnp.cos(theta) * v_scale
                vy = jnp.sin(theta) * v_scale
                vx = jnp.where(spawn_left, -vx, vx)
                x_pos = jnp.where(spawn_left, -radius, WORLD_DIM + radius)

                theme = jnp.where(is_flyer, flyer_theme, theme_pick)
                # ``kind`` is traced, so pick the sprite aspect with a select chain.
                aspect = jnp.select(
                    [kind == METEOR, kind == CLOUD, kind == TURRET, is_flyer],
                    [
                        ASPECTS[METEOR][jnp.clip(theme, 0, len(METEOR_SPRITES) - 1)],
                        ASPECTS[CLOUD][jnp.clip(theme, 0, len(CLOUD_SPRITES) - 1)],
                        ASPECTS[TURRET][jnp.clip(theme, 0, len(TURRET_SPRITES) - 1)],
                        ASPECTS[FLYER][jnp.clip(theme, 0, len(FLYER_SPRITES) - 1)],
                    ],
                    1.0,
                )
                ry = jnp.where(kind == TURRET, radius / aspect, radius)
                entry = engine.entity(
                    x_pos,
                    y_pos,
                    vx,
                    vy,
                    rx=radius,
                    ry=ry,
                    type=kind,
                    image_type=kind,
                    image_theme=theme,
                    is_reflected=spawn_left,
                    spawn_time=t + unit * 5,
                    fire_time=fire_time,
                    health=HEALTH[kind],
                    active=active,
                )
                slot = jnp.clip(count, 0, MAX_SPAWNERS - 1)
                pool = jax.tree.map(lambda buf, value, slot=slot: buf.at[slot].set(value), pool, entry)
                count = count + 1
                return state, pool, count

            state, pool, count = jax.lax.fori_loop(0, MAX_GROUP_SIZE, unit_body, (state, pool, count))

            state, delta = jax.lax.cond(
                active,
                lambda s: rng.randint(s, MIN_DELTA_T, MAX_DELTA_T),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            t = jnp.where(active, t + delta, t)
            return state, pool, count, t

        state, pool, count, _ = jax.lax.fori_loop(0, MAX_TIMELINE_STEPS, round_body, (state, pool, count, t))
        # Only the units that were actually written belong to the timeline.
        filled = jnp.arange(MAX_SPAWNERS) < jnp.minimum(count, MAX_SPAWNERS)
        blank = empty_pool(MAX_SPAWNERS)
        pool = jax.tree.map(lambda buf, zero: jnp.where(filled, buf, zero), pool, blank)
        # ``std::sort(spawners, spawn_cmp)``: latest first.
        order = jnp.argsort(-pool.spawn_time.astype(jnp.int32))
        return jax.tree.map(lambda field: field[order], pool), state

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: StarPilotState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy, is_reflected=action_vx < 0)

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(-1),
            is_blocked=lambda tile, is_horizontal: tile == -1,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )
        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- bullets hit enemies -------------------------------------------- #
        def hit(index, carry):
            pool, reward, agent = carry
            bullet = jax.tree.map(lambda field: field[index], pool)
            is_bullet = pool.collides_with_entities[index] & pool.active[index]
            target = engine.collides(bullet, pool) & pool.active & is_bullet
            target = target.at[index].set(False)
            not_cloud = pool.type != CLOUD
            destructive = (
                (pool.type == FLYER) | (pool.type == FAST_FLYER) | (pool.type == TURRET) | (pool.type == METEOR)
            )
            hit_something = (target & not_cloud & destructive).any()
            pool = pool._replace(active=pool.active.at[index].set(pool.active[index] & ~hit_something))
            health = pool.health - jnp.where(target & not_cloud & destructive, 1.0, 0.0)
            pool = pool._replace(health=health)
            return pool, reward, agent

        pool, reward, agent = jax.lax.fori_loop(0, MAX_ENTITIES, hit, (pool, jnp.zeros((), jnp.float32), agent))

        # ---- enemies fire and die ------------------------------------------- #
        def fire_and_reap(index, carry):
            pool, reward, agent = carry
            ent = jax.tree.map(lambda field: field[index], pool)
            alive = ent.active & (ent.type != engine.PLAYER)
            turret = ent.type == TURRET
            elapsed = now - ent.spawn_time
            should_fire = (
                alive
                & (ent.fire_time > 0)
                & jnp.where(turret, (elapsed % jnp.maximum(ent.fire_time, 1)) == 0, elapsed == ent.fire_time)
            )
            bullet_type = jnp.where(turret, BULLET3, BULLET2)
            dx = agent.x - ent.x
            dy = agent.y - ent.y
            dist = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-6)
            bv = VS[bullet_type] * V_SCALE / dist
            bullet = engine.entity(
                ent.x,
                ent.y,
                dx * bv,
                dy * bv,
                rx=PLAYER_BULLET_RADIUS,
                ry=PLAYER_BULLET_RADIUS,
                type=bullet_type,
                image_type=bullet_type,
                active=should_fire,
            )
            free = ~pool.active
            slot = jnp.argmax(free)
            write = should_fire & free.any()
            pool = jax.lax.cond(
                write,
                lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, bullet),
                lambda p: p,
                pool,
            )

            destructible = (ent.type == FLYER) | (ent.type == FAST_FLYER) | (ent.type == TURRET) | (ent.type == METEOR)
            dead = alive & destructible & (ent.health <= 0)
            pool = pool._replace(active=pool.active.at[index].set(pool.active[index] & ~dead))
            return pool, reward + jnp.where(dead, ENEMY_REWARD, 0.0), agent

        pool, reward, agent = jax.lax.fori_loop(
            0, MAX_ENTITIES, fire_and_reap, (pool, jnp.zeros((), jnp.float32), agent)
        )

        # ---- spawn the timeline entries due this step ----------------------- #
        due = env_state.spawners.active & (env_state.spawners.spawn_time == now)
        slots = jnp.cumsum(due.astype(jnp.int32)) - 1 + jnp.argmax(~pool.active)
        free_count = jnp.sum(~pool.active)
        order = jnp.argsort(jnp.where(due, jnp.arange(MAX_SPAWNERS), MAX_SPAWNERS))
        writes = jnp.minimum(jnp.sum(due), free_count)

        def spawn_one(step, carry):
            pool = carry
            source = order[step]
            slot = jnp.argmax(~pool.active)
            entry = jax.tree.map(lambda field: field[source], env_state.spawners)
            active = step < writes
            pool = jax.lax.cond(
                active,
                lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, entry),
                lambda p: p,
                pool,
            )
            return pool

        pool = jax.lax.fori_loop(0, MAX_SPAWNERS, spawn_one, pool)
        del slots

        # ---- the player's gun ------------------------------------------------ #
        firing = special_action != 0
        theta = jnp.where(special_action == 2, jnp.pi, 0.0)
        bullet_speed = VS[BULLET_PLAYER] * V_SCALE
        shot = engine.entity(
            agent.x + agent.rx * jnp.cos(theta),
            agent.y,
            jnp.cos(theta) * bullet_speed,
            jnp.sin(theta) * bullet_speed,
            rx=PLAYER_BULLET_RADIUS,
            ry=PLAYER_BULLET_RADIUS,
            type=BULLET_PLAYER,
            image_type=BULLET_PLAYER,
            collides_with_entities=True,
            active=firing,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            firing & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, shot),
            lambda p: p,
            pool,
        )

        # ---- the finish line at step 500 ------------------------------------- #
        launch = now == SHOOTER_WIN_TIME
        finish = engine.entity(
            WORLD_DIM + 2.0,
            WORLD_DIM / 2,
            -SLOW_V * V_SCALE,
            0.0,
            rx=2.0,
            ry=WORLD_DIM / 2,
            type=FINISH_LINE,
            image_type=FINISH_LINE,
            active=launch,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            launch & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, finish),
            lambda p: p,
            pool,
        )

        # ---- contact with the agent ------------------------------------------ #
        touching = engine.collides(agent, pool) & pool.active
        lethal = (
            (pool.type == FLYER)
            | (pool.type == FAST_FLYER)
            | (pool.type == BULLET2)
            | (pool.type == BULLET3)
            | (pool.type == TURRET)
            | (pool.type == METEOR)
        )
        reached_finish = (touching & (pool.type == FINISH_LINE)).any()
        killed = (touching & lethal).any()
        terminated = reached_finish | killed | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)
        reward = reward + jnp.where(reached_finish, COMPLETION_BONUS, 0.0)

        stepped = StarPilotState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            spawners=env_state.spawners,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: StarPilotState,
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

    def observe(self, env_state: StarPilotState) -> jax.Array:
        """The whole 16x16 arena: ``center_agent`` is off."""
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
            background=SPACE_COLOUR,
        )
