"""A pure-JAX port of ProcGen's ``caveflyer``, from ``procgen/src/games/caveflyer.cpp``.

A lunar lander in a cave: the ship thrusts along its facing, which the horizontal
axis of the action turns, and touching anything but the green saucer ends the run.
Red saucers take five hits and pay three; the green one pays ten.

The cave is another cellular-automata level: random fill, four smoothing passes, the
largest room, then the path to the goal widened and the cave re-smoothed around it.

Deviations: the exhaust puff and the bullet/impact explosions are not spawned (both
are decoration), and sprites are not rotated on screen even though the ship's
``rotation`` drives the physics.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, roomgen, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``caveflyer.cpp`` object ids.
GOAL = 1
OBSTACLE = 2
TARGET = 3
PLAYER_BULLET = 4
ENEMY = 5
CAVEWALL = 8
#: A temporary placement marker, kept out of the wall id space.
MARKER_TILE = 11
#: ProcGen's own ids, which the room generator speaks; the cave stays in that id
#: space until the end, where the walls become ``CAVEWALL``.
MAZE_WALL = 51
SPACE = 100

GOAL_REWARD = 10.0
TARGET_REWARD = 3.0
TARGET_HEALTH = 5.0
WORLD_DIM = 40
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.9
MAX_SPEED = 0.5
MIXRATE_ROT = 0.5
MAX_V_THETA = 15 * jnp.pi / 180
AGENT_RADIUS = 0.4
ORB_RADIUS = 0.5
BULLET_RX = 0.1
BULLET_RY = 0.25
BULLET_EXPIRE = 10
WIDEN_ROUNDS = 4
SMOOTH_PASSES = 4
#: ``chunk = free // 80`` and three chunks of objects, so at most sixty.
MAX_OBJECTS = 60
MAX_ENTITIES = 128
MAX_SUB_STEPS = 6

TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32).at[CAVEWALL].set(sprites.row('misc/caveflyer/groundA'))
)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[GOAL, 0]
    .set(sprites.row('misc/caveflyer/ufoGreen2'))
    .at[OBSTACLE, 0]
    .set(sprites.row('misc/caveflyer/meteorBrown_big1'))
    .at[TARGET, 0]
    .set(sprites.row('misc/caveflyer/ufoRed2'))
    .at[PLAYER_BULLET, 0]
    .set(sprites.row('misc/caveflyer/laserBlue02'))
    .at[ENEMY, 0]
    .set(sprites.row('misc/caveflyer/enemyShipBlue4'))
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/caveflyer/playerShip1_red'))
)
#: The original paints a space background; this is a flat stand-in.
SPACE_COLOUR = (12, 12, 22)


class CaveFlyerState(NamedTuple):
    """Game state: the cave, the ship (with its rotation) and everything in it."""

    grid: jax.Array  # (WORLD_DIM * WORLD_DIM,) CAVEWALL or SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) goal, obstacles, saucers, enemies, lasers
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def _simple_choose(state, size, count, active):
    """``simple_choose`` with a per-pick mask: inactive picks consume nothing."""
    chosen = jnp.zeros((count,), jnp.int32)
    taken = jnp.zeros((WORLD_DIM * WORLD_DIM,), bool)
    limit = WORLD_DIM * WORLD_DIM - 1
    for pick in range(count):
        live = active[pick]

        high = jnp.maximum(size, 1)

        def body(carry, live=live, taken_at=taken, limit_high=high):
            state, _, _ = carry
            state, draw = jax.lax.cond(
                live,
                lambda st: rng.randn(st, limit_high),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            return state, draw, live & taken_at[jnp.clip(draw, 0, limit)]

        state, value, _ = jax.lax.while_loop(lambda carry: carry[2], body, (state, jnp.zeros((), jnp.int32), live))
        chosen = chosen.at[pick].set(jnp.where(live, value, 0))
        taken = taken.at[jnp.clip(value, 0, limit)].set(taken[jnp.clip(value, 0, limit)] | live)
    return state, chosen


class CaveFlyerEnv(Env):
    """ProcGen's ``caveflyer`` as a functional, jit-friendly environment."""

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

        def reset(env_state: CaveFlyerState) -> CaveFlyerState:
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
    def reset_env_state(self, key: jax.Array) -> CaveFlyerState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        size = WORLD_DIM * WORLD_DIM
        cells = jnp.arange(size, dtype=jnp.int32)

        # ---- the noisy cave, smoothed four times ----------------------------- #
        def fill(index, carry):
            state, flat = carry
            state, draw = rng.rand01(state)
            return state, flat.at[index].set(jnp.where(draw < 0.5, MAZE_WALL, SPACE))

        state, flat = jax.lax.fori_loop(0, size, fill, (state, jnp.full((size,), SPACE, jnp.int32)))
        for _ in range(SMOOTH_PASSES):
            flat = roomgen.update(flat, WORLD_DIM)

        best_room = roomgen.find_best_room(flat, WORLD_DIM)
        flat = jnp.where(best_room, SPACE, MAZE_WALL)

        # ---- the ship, the saucer and the tunnel between them ----------------- #
        free_mask = flat == SPACE
        free_count = free_mask.sum()
        free_cells = jnp.sort(jnp.where(free_mask, cells, size))
        state, picks = _simple_choose(state, free_count, 2, jnp.ones((2,), bool))
        agent_cell = free_cells[jnp.clip(picks[0], 0, size - 1)]
        goal_cell = free_cells[jnp.clip(picks[1], 0, size - 1)]

        path_cells, path_length = roomgen.find_path(flat, WORLD_DIM, agent_cell, goal_cell)
        on_path = jnp.arange(path_cells.shape[0]) < path_length
        wide = jnp.zeros((size + 1,), bool).at[jnp.where(on_path, path_cells, size)].set(True)[:size]
        wide = roomgen.expand_room(flat, WORLD_DIM, wide, WIDEN_ROUNDS)
        flat = jnp.where(wide, SPACE, MAZE_WALL)

        # the cave keeps smoothing, but the shipped tunnel stays open
        for _ in range(SMOOTH_PASSES):
            flat = roomgen.update(flat, WORLD_DIM)
            flat = jnp.where(on_path_scatter(on_path, path_cells, size), SPACE, flat)

        flat = jnp.where(on_path_scatter(on_path, path_cells, size), MARKER_TILE, flat)

        # ---- obstacles, saucers and ships ------------------------------------ #
        open_mask = flat == SPACE
        free_count = open_mask.sum()
        free_cells = jnp.sort(jnp.where(open_mask, cells, size))
        chunk = free_count // 80
        num_objects = 3 * chunk
        active = jnp.arange(MAX_OBJECTS) < num_objects
        state, object_picks = _simple_choose(state, free_count, MAX_OBJECTS, active)
        object_cells = free_cells[jnp.clip(object_picks, 0, size - 1)]

        walls = jnp.where(flat == MAZE_WALL, CAVEWALL, flat)
        walls = jnp.where(walls == MARKER_TILE, SPACE, walls)

        pool = empty_pool()
        slot = jnp.zeros((), jnp.int32)

        def write(pool, slot, ent, live):
            position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=position, on=live: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                ent,
            )
            return pool, slot + live

        goal = engine.entity(
            (goal_cell % WORLD_DIM).astype(jnp.float32) + 0.5,
            (goal_cell // WORLD_DIM).astype(jnp.float32) + 0.5,
            rx=ORB_RADIUS,
            ry=ORB_RADIUS,
            type=GOAL,
            image_type=GOAL,
            collides_with_entities=True,
        )
        pool, slot = write(pool, slot, goal, jnp.asarray(True))

        def place_object(index, carry):
            state, pool, slot = carry
            cell = jnp.clip(object_cells[index], 0, size - 1)
            enemy_slot = active[index] & (index >= 2 * chunk)
            state, speed_draw = jax.lax.cond(
                enemy_slot, lambda st: rng.rand01(st), lambda st: (st, jnp.zeros((), jnp.float32)), state
            )
            state, sign_draw = jax.lax.cond(
                enemy_slot, lambda st: rng.randn(st, 2), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            state, axis_draw = jax.lax.cond(
                enemy_slot, lambda st: rng.rand01(st), lambda st: (st, jnp.zeros((), jnp.float32)), state
            )
            velocity = (0.1 * speed_draw + 0.1) * (sign_draw * 2 - 1)
            kind = jnp.where(index < chunk, OBSTACLE, jnp.where(index < 2 * chunk, TARGET, ENEMY))
            body = engine.entity(
                (cell % WORLD_DIM).astype(jnp.float32) + 0.5,
                (cell // WORLD_DIM).astype(jnp.float32) + 0.5,
                jnp.where(axis_draw < 0.5, velocity, 0.0),
                jnp.where(axis_draw < 0.5, 0.0, velocity),
                rx=ORB_RADIUS,
                ry=ORB_RADIUS,
                type=kind,
                image_type=kind,
                health=jnp.where(kind == TARGET, TARGET_HEALTH, 0.0),
                smart_step=kind == ENEMY,
                collides_with_entities=True,
                active=active[index],
            )
            return write_pool(state, pool, slot, body, active[index])

        def write_pool(state, pool, slot, ent, live):
            position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=position, on=live: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                ent,
            )
            return state, pool, slot + live

        state, pool, slot = jax.lax.fori_loop(0, MAX_OBJECTS, place_object, (state, pool, slot))

        agent = engine.entity(
            (agent_cell % WORLD_DIM).astype(jnp.float32) + 0.5,
            (agent_cell // WORLD_DIM).astype(jnp.float32) + 0.5,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )
        return CaveFlyerState(
            grid=walls,
            agent=agent,
            entities=pool,
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: CaveFlyerState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        grid = env_state.grid

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)

        # ---- set_action_xy: thrust along the facing, and a turn --------------- #
        acceleration = move_action % 3 - 1
        acceleration = jnp.where(acceleration < 0, acceleration * 0.33, acceleration)
        theta = -env_state.agent.rotation + jnp.pi / 2
        action_vx = acceleration * jnp.cos(theta)
        action_vy = acceleration * jnp.sin(theta)
        action_vrot = move_action // 3 - 1

        # ---- update_agent_velocity (caveflyer's, which only adds thrust) ------ #
        agent = env_state.agent
        vx = (agent.vx + MIXRATE * MAX_SPEED * action_vx * 0.2) * 0.9
        vy = (agent.vy + MIXRATE * MAX_SPEED * action_vy * 0.2) * 0.9
        vrot = MIXRATE_ROT * agent.vrot + MIXRATE_ROT * MAX_V_THETA * action_vrot
        agent = agent._replace(vx=vx, vy=vy, vrot=vrot, is_reflected=action_vrot < 0)

        agent, _ = engine.step_object(
            agent,
            grid.reshape(WORLD_DIM, WORLD_DIM),
            out_of_bounds=jnp.asarray(CAVEWALL),
            is_blocked=lambda tile, is_horizontal: tile == CAVEWALL,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )
        # ``Entity::step`` turns every body, the ship included
        agent = agent._replace(rotation=agent.rotation + agent.vrot)

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- firing ----------------------------------------------------------- #
        firing = special_action == 1
        laser = engine.entity(
            agent.x,
            agent.y,
            jnp.cos(theta),
            jnp.sin(theta),
            rx=BULLET_RX,
            ry=BULLET_RY,
            type=PLAYER_BULLET,
            image_type=PLAYER_BULLET,
            rotation=agent.rotation,
            expire_time=BULLET_EXPIRE,
            active=firing,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            firing & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, laser),
            lambda p: p,
            pool,
        )

        # ---- lasers that touch a cave wall die -------------------------------- #
        view = jnp.pad(grid.reshape(WORLD_DIM, WORLD_DIM), 1, constant_values=CAVEWALL)
        is_laser = pool.active & (pool.type == PLAYER_BULLET)
        corners = []
        for i in (0, 1):
            for j in (0, 1):
                cx = jnp.clip((pool.x + pool.rx * (2 * i - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                cy = jnp.clip((pool.y + pool.ry * (2 * j - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                corners.append(view[cy + 1, cx + 1] == CAVEWALL)
        in_wall = jnp.stack(corners).any(axis=0) & is_laser
        pool = pool._replace(active=pool.active & ~in_wall)

        # ---- laser hits ------------------------------------------------------- #
        def laser_hits(index, carry):
            pool, reward = carry
            laser_body = jax.tree.map(lambda field: field[index], pool)
            is_shot = pool.active[index] & (pool.type[index] == PLAYER_BULLET)
            hit = engine.collides(laser_body, pool) & pool.active & is_shot
            hit = hit.at[index].set(False)
            struck = hit & (pool.type != PLAYER_BULLET)
            target_hit = struck & (pool.type == TARGET)
            pool = pool._replace(health=pool.health - jnp.where(target_hit, 1.0, 0.0))
            killed = target_hit & (pool.health <= 0)
            pool = pool._replace(active=pool.active & ~killed)
            reward = reward + TARGET_REWARD * killed.sum()
            pool = pool._replace(active=pool.active.at[index].set(pool.active[index] & ~struck.any()))
            return pool, reward

        pool, reward = jax.lax.fori_loop(0, MAX_ENTITIES, laser_hits, (pool, jnp.zeros((), jnp.float32)))

        # ---- contact with the ship --------------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        reached_goal = (touching & (pool.type == GOAL)).any()
        hit_anything = touching & ((pool.type == OBSTACLE) | (pool.type == ENEMY) | (pool.type == TARGET))
        reward = reward + jnp.where(reached_goal, GOAL_REWARD, 0.0)
        terminated = reached_goal | hit_anything.any()

        stepped = CaveFlyerState(
            grid=grid,
            agent=agent,
            entities=pool,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: CaveFlyerState,
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

    def observe(self, env_state: CaveFlyerState) -> jax.Array:
        """``visibility = 16`` around the ship."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        return render_textured(
            env_state.grid.reshape(WORLD_DIM, WORLD_DIM),
            jnp.stack([env_state.agent.x, env_state.agent.y]),
            bodies,
            TILE_ROWS,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            16,
            jnp.asarray(CAVEWALL),
            background=SPACE_COLOUR,
        )


def on_path_scatter(on_path, path_cells, size):
    """A flat mask of the cells on the path."""
    return jnp.zeros((size + 1,), bool).at[jnp.where(on_path, path_cells, size)].set(True)[:size]
