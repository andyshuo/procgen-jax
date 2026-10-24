"""A pure-JAX port of ProcGen's ``dodgeball``, from ``procgen/src/games/dodgeball.cpp``.

Arena dodgeball: the level is a rectangle cut into rooms by lava walls, you throw
balls (action 9, seven-step cooldown) at enemies who throw them back, and the exit
door opens once the last enemy is down.

The generator is a recursive split: pick a random room, cut it in two with a lava
wall, and keep the pieces that are still big enough -- four times, then drop the
door on a random wall and scatter the enemies.

Deviations: the dust cloud an enemy leaves is not spawned (visual only) and the
top-down background is flat.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``dodgeball.cpp`` object ids.
LAVA_WALL = 1
PLAYER_BALL = 3
ENEMY = 4
DOOR = 5
ENEMY_BALL = 6
DOOR_OPEN = 7
DUST_CLOUD = 8
OOB_WALL = 10

COMPLETION_BONUS = 10.0
ENEMY_REWARD = 2.0
NUM_ENEMY_THEMES = 7
ENEMY_VEL = 0.05
#: ``BALL_V_ROT = PI * 0.23`` only spins the sprite.
WORLD_DIM = 20
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.5
#: HardMode's numbers from ``game_reset``.
NUM_ITERATIONS = 4
THICKNESS = 0.3 * 1.5
ENEMY_RADIUS = 0.5 * 1.5
BALL_RADIUS = 0.25 * 1.5
BALL_VSCALE = 0.25 * 1.5
EXIT_RADIUS = 0.75
AGENT_RADIUS = 0.75
MAX_EXTRA_ENEMIES = 3
ENEMY_FIRE_DELAY = 50
FIRE_COOLDOWN = 7
BALL_EXPIRE = 50
#: ``hard_min_dim = 4 * agent->rx + 2 * thickness + .5``, ``min_dim = agent->rx * 8 + .5``.
HARD_MIN_DIM = 4 * AGENT_RADIUS + 2 * THICKNESS + 0.5
MIN_DIM = AGENT_RADIUS * 8 + 0.5
#: One room becomes up to three, four times over.
MAX_ROOMS = 16
MAX_ENTITIES = 48
MAX_SUB_STEPS = 6
#: ``spawn_entities`` is asked for three to six enemies.
MAX_SPAWNS = MAX_EXTRA_ENEMIES + 3
REPOSITION_ATTEMPTS = 100

ENEMY_SPRITES = [f'character{index}' for index in range(1, 12)]

TILE_SPRITES = jnp.full((64,), sprites.row('_transparent'), jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 12), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/dodgeball/character12'))
    .at[PLAYER_BALL, 0]
    .set(sprites.row('misc/dodgeball/ball_soccer1'))
    .at[ENEMY_BALL, 0]
    .set(sprites.row('misc/dodgeball/ball_soccer2'))
    .at[DOOR, 0]
    .set(sprites.row('misc/dodgeball/blockRed'))
    .at[DOOR_OPEN, 0]
    .set(sprites.row('misc/dodgeball/blockGreen'))
    .at[LAVA_WALL, 0]
    .set(sprites.row('misc/dodgeball/tileStone_slope2'))
)
for _index, _name in enumerate(ENEMY_SPRITES[:NUM_ENEMY_THEMES]):
    ENTITY_SPRITES = ENTITY_SPRITES.at[ENEMY, _index].set(sprites.row(f'misc/dodgeball/{_name}'))
#: The original paints a top-down floor image; this is a flat stand-in.
FLOOR_COLOUR = (52, 44, 40)


class DodgeballState(NamedTuple):
    """Game state: the agent, the walls, the enemies and the balls."""

    grid: jax.Array  # (WORLD_DIM, WORLD_DIM), all SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) walls, enemies, the door and balls
    num_enemies: jax.Array
    last_fire_time: jax.Array
    last_move_action: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class DodgeballEnv(Env):
    """ProcGen's ``dodgeball`` as a functional, jit-friendly environment."""

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

        def reset(env_state: DodgeballState) -> DodgeballState:
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
    def _rand_pos(self, state, radius_x, radius_y, low_x, low_y, size_x, size_y, active):
        """``rand_pos`` on both axes: one draw each unless the span is too tight."""
        draws_x = (size_x) > 2 * radius_x
        draws_y = (size_y) > 2 * radius_y
        state, draw_x = jax.lax.cond(
            draws_x & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        state, draw_y = jax.lax.cond(
            draws_y & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        x = jnp.where(draws_x, (size_x - 2 * radius_x) * draw_x + radius_x + low_x, (low_x + size_x + low_x) / 2)
        y = jnp.where(draws_y, (size_y - 2 * radius_y) * draw_y + radius_y + low_y, (low_y + size_y + low_y) / 2)
        return state, x, y

    def _reposition(self, state, pool, agent, radius_x, radius_y, low_x, low_y, size_x, size_y, active):
        """``reposition``: retry inside the box until the body is clear."""
        start = (
            state,
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.asarray(False),
        )

        def attempt(_, carry):
            state, x, y, done = carry
            state, nx, ny = self._rand_pos(state, radius_x, radius_y, low_x, low_y, size_x, size_y, active & ~done)
            candidate = engine.entity(nx, ny, rx=radius_x, ry=radius_y)
            blocked = (engine.collides(candidate, pool) & pool.active).any() | engine.collides(candidate, agent)
            fresh = active & ~done & ~blocked
            return (
                state,
                jnp.where(fresh, nx, x),
                jnp.where(fresh, ny, y),
                done | fresh,
            )

        return jax.lax.fori_loop(0, REPOSITION_ATTEMPTS, attempt, start)

    def _choose_vel(self, state):
        """``choose_vel``: three draws -- speed, axis, and the idle timer."""
        state, speed_draw = rng.randn(state, 2)
        state, axis_draw = rng.randn(state, 2)
        state, timer_draw = rng.randn(state, 50)
        vel = ENEMY_VEL * (speed_draw * 2 - 1)
        on_x = axis_draw == 0
        return (
            state,
            jnp.where(on_x, vel, 0.0),
            jnp.where(on_x, 0.0, vel),
            timer_draw + 25,
        )

    def reset_env_state(self, key: jax.Array) -> DodgeballState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        pool = empty_pool()
        n_entities = jnp.zeros((), jnp.int32)

        # ---- split the arena ------------------------------------------------- #
        rooms = (
            jnp.zeros((MAX_ROOMS, 4), jnp.float32)
            .at[0]
            .set(jnp.asarray([0.0, 0.0, float(WORLD_DIM), float(WORLD_DIM)]))
        )
        count = jnp.asarray(1, jnp.int32)
        slots = jnp.arange(MAX_ROOMS)

        for _ in range(NUM_ITERATIONS):
            active = slots < count
            room_count = jnp.maximum(count, 1)
            state, room_draw = jax.lax.cond(
                active.any(),
                lambda s, high=room_count: rng.randn(s, high),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            rank = jnp.cumsum(active.astype(jnp.int32)) - 1
            slot = jnp.argmax(active & (rank == room_draw))
            chosen = rooms[slot]
            # ``rooms.erase(begin() + idx)`` keeps the relative order.
            shifted = jnp.concatenate([rooms[1:], jnp.zeros((1, 4), jnp.float32)])
            keep = slots < slot
            rooms = jnp.where(keep[:, None], rooms, shifted)
            count = count - 1

            rw, rh = chosen[2], chosen[3]
            state, width_draw = rng.rand01(state)
            state, choice_draw = rng.rand01(state)
            state, gap_draw = rng.randn(state, 3)
            will_split_width = width_draw < 0.5
            will_split_width = jnp.where(rw < MIN_DIM, False, will_split_width)
            will_split_width = jnp.where(rh < MIN_DIM, True, will_split_width)
            choice2 = choice_draw < 0.5
            gap = 0.25 * (gap_draw + 1)
            pct = 1 - gap

            # horizontal wall: split the height
            wy = jnp.where(choice2, chosen[1], chosen[1] + (1 - pct) * rh)
            remy = jnp.where(choice2, chosen[1] + pct * rh, chosen[1])
            wh = pct * rh
            nextw = rw / 2 - THICKNESS
            room_a = jnp.stack([chosen[0], wy, nextw, wh])
            room_b = jnp.stack([chosen[0] + rw / 2 + THICKNESS, wy, nextw, wh])
            room_c = jnp.stack([chosen[0], remy, rw, rh - wh])
            wall_x = chosen[0] + rw / 2
            wall_y = wy + wh / 2
            wall_rx = THICKNESS
            wall_ry = wh / 2

            # vertical wall: split the width
            wx = jnp.where(choice2, chosen[0], chosen[0] + (1 - pct) * rw)
            remx = jnp.where(choice2, chosen[0] + pct * rw, chosen[0])
            ww = pct * rw
            nexth = rh / 2 - THICKNESS
            room_a = jnp.where(will_split_width, jnp.stack([wx, chosen[1], ww, nexth]), room_a)
            room_b = jnp.where(will_split_width, jnp.stack([wx, chosen[1] + rh / 2 + THICKNESS, ww, nexth]), room_b)
            room_c = jnp.where(will_split_width, jnp.stack([remx, chosen[1], rw - ww, rh]), room_c)
            wall_x = jnp.where(will_split_width, wx + ww / 2, wall_x)
            wall_y = jnp.where(will_split_width, chosen[1] + rh / 2, wall_y)
            wall_rx = jnp.where(will_split_width, ww / 2, wall_rx)
            wall_ry = jnp.where(will_split_width, THICKNESS, wall_ry)

            wall = engine.entity(
                wall_x,
                wall_y,
                rx=wall_rx,
                ry=wall_ry,
                type=LAVA_WALL,
                image_type=LAVA_WALL,
                active=True,
            )
            slot_now = jnp.clip(n_entities, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(lambda buf, value, s=slot_now: buf.at[s].set(value), pool, wall)
            n_entities = n_entities + 1

            def fits(room):
                return (
                    ((room[2] >= MIN_DIM) | (room[3] >= MIN_DIM))
                    & (room[2] >= HARD_MIN_DIM)
                    & (room[3] >= HARD_MIN_DIM)
                )

            for room in (room_a, room_b, room_c):
                room = jnp.asarray(room, jnp.float32)
                ok = fits(room)
                slot_now = jnp.clip(count, 0, MAX_ROOMS - 1)
                rooms = rooms.at[slot_now].set(jnp.where(ok, room, rooms[slot_now]))
                count = count + ok

        # ---- the exit door on one of the four walls --------------------------- #
        door_len = 2 * EXIT_RADIUS
        state, wall_choice = rng.randn(state, 4)
        boxes = jnp.stack(
            [
                jnp.asarray([0.0, 0.0, float(WORLD_DIM), door_len]),
                jnp.asarray([0.0, WORLD_DIM - door_len, float(WORLD_DIM), door_len]),
                jnp.asarray([0.0, 0.0, door_len, float(WORLD_DIM)]),
                jnp.asarray([WORLD_DIM - door_len, 0.0, door_len, float(WORLD_DIM)]),
            ]
        )
        box = boxes[wall_choice]
        agent_placeholder = engine.entity(-1e3, -1e3, rx=AGENT_RADIUS, ry=AGENT_RADIUS)
        state, door_x, door_y, _ = self._reposition(
            state,
            pool,
            agent_placeholder,
            EXIT_RADIUS,
            EXIT_RADIUS,
            box[0],
            box[1],
            box[2],
            box[3],
            jnp.asarray(True),
        )
        door = engine.entity(
            door_x,
            door_y,
            rx=EXIT_RADIUS,
            ry=EXIT_RADIUS,
            type=DOOR,
            image_type=DOOR,
            active=True,
        )

        # ---- the agent, then the enemies ------------------------------------- #
        agent = engine.entity(
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )

        slot_now = jnp.clip(n_entities, 0, MAX_ENTITIES - 1)
        pool = jax.tree.map(lambda buf, value, s=slot_now: buf.at[s].set(value), pool, door)
        n_entities = n_entities + 1

        # ``reposition_agent``: the agent is placed before the enemies exist.
        state, agent_x, agent_y, _ = self._reposition(
            state,
            pool,
            engine.entity(-1e3, -1e3, rx=AGENT_RADIUS, ry=AGENT_RADIUS),
            AGENT_RADIUS,
            AGENT_RADIUS,
            0.0,
            0.0,
            float(WORLD_DIM),
            float(WORLD_DIM),
            jnp.asarray(True),
        )
        agent = agent._replace(x=agent_x, y=agent_y)

        state, enemy_count_draw = rng.randn(state, MAX_EXTRA_ENEMIES + 1)
        num_enemies = enemy_count_draw + 3

        for spawn_index in range(MAX_SPAWNS):
            spawn = spawn_index < num_enemies
            state, x, y, _ = self._reposition(
                state,
                pool,
                agent,
                ENEMY_RADIUS,
                ENEMY_RADIUS,
                0.0,
                0.0,
                float(WORLD_DIM),
                float(WORLD_DIM),
                spawn,
            )
            enemy = engine.entity(
                x,
                y,
                rx=ENEMY_RADIUS,
                ry=ENEMY_RADIUS,
                type=ENEMY,
                image_type=ENEMY,
                active=spawn,
            )
            slot_now = jnp.clip(n_entities, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=slot_now, write=spawn: buf.at[s].set(jnp.where(write, value, buf[s])),
                pool,
                enemy,
            )
            n_entities = n_entities + spawn

        # ---- dress the entities ---------------------------------------------- #
        state, enemy_theme = rng.randn(state, NUM_ENEMY_THEMES)
        num_enemies = num_enemies.astype(jnp.int32)
        enemies = pool.type == ENEMY
        pool = pool._replace(
            image_theme=jnp.where(enemies, enemy_theme, pool.image_theme),
            health=jnp.where(enemies, 1.0, pool.health),
            spawn_time=jnp.where(enemies, 0, pool.spawn_time),
            fire_time=jnp.where(enemies, 10, pool.fire_time),
            collides_with_entities=enemies | (pool.type == LAVA_WALL),
            smart_step=enemies,
        )
        # ``choose_vel`` runs once per enemy, in entity order.
        for index in range(MAX_ENTITIES):
            is_enemy = enemies[index] & pool.active[index]
            state, vel_x, vel_y, timer = self._choose_vel(state)
            pool = pool._replace(
                vx=jnp.where(is_enemy, vel_x, pool.vx),
                vy=jnp.where(is_enemy, vel_y, pool.vy),
                spawn_time=jnp.where(is_enemy, timer, pool.spawn_time),
            )

        return DodgeballState(
            grid=jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32),
            agent=agent,
            entities=pool,
            num_enemies=num_enemies,
            last_fire_time=jnp.zeros((), jnp.int32),
            last_move_action=jnp.asarray(7, jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: DodgeballState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)
        last_move_action = jnp.where(move_action != 4, move_action, env_state.last_move_action)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy, is_reflected=action_vx < 0)

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(OOB_WALL),
            is_blocked=lambda tile, is_horizontal: tile == OOB_WALL,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = env_state.entities
        state = env_state.rng

        # ---- the player throws a ball ---------------------------------------- #
        can_fire = (special_action == 1) & ((now - env_state.last_fire_time) >= FIRE_COOLDOWN)
        throw_vx = (last_move_action // 3 - 1) * BALL_VSCALE
        throw_vy = (last_move_action % 3 - 1) * BALL_VSCALE
        ball = engine.entity(
            agent.x,
            agent.y,
            throw_vx,
            throw_vy,
            rx=BALL_RADIUS,
            ry=BALL_RADIUS,
            type=PLAYER_BALL,
            image_type=PLAYER_BALL,
            collides_with_entities=True,
            expire_time=BALL_EXPIRE,
            active=can_fire,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            can_fire & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, ball),
            lambda p: p,
            pool,
        )
        last_fire_time = jnp.where(can_fire, now, env_state.last_fire_time)

        # ---- enemies wander and throw ---------------------------------------- #
        def enemy_turn(index, carry):
            pool, state, agent = carry
            ent = jax.tree.map(lambda field: field[index], pool)
            is_enemy = ent.active & (ent.type == ENEMY)
            state, vel_x, vel_y, timer = self._choose_vel(state)
            respawn = is_enemy & (ent.spawn_time == 0)
            pool = pool._replace(
                vx=jnp.where(respawn, vel_x, pool.vx),
                vy=jnp.where(respawn, vel_y, pool.vy),
                spawn_time=jnp.where(respawn, timer, pool.spawn_time - 1),
            )

            can_fire = is_enemy & ((now - ent.fire_time) >= ENEMY_FIRE_DELAY)
            dx = ent.x - agent.x
            dy = ent.y - agent.y
            toward_x = jnp.where(ent.x < agent.x, 1.0, -1.0)
            toward_y = jnp.where(ent.y < agent.y, 1.0, -1.0)
            aligned_x = can_fire & (jnp.abs(dx) < 1)
            aligned_y = can_fire & (jnp.abs(dy) < 1)
            state, cool = jax.lax.cond(
                aligned_x | aligned_y,
                lambda s: rng.randn(s, 4),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            thrown = engine.entity(
                ent.x,
                ent.y,
                jnp.where(aligned_y, toward_x, 0.0) * BALL_VSCALE,
                jnp.where(aligned_x, toward_y, 0.0) * BALL_VSCALE,
                rx=BALL_RADIUS,
                ry=BALL_RADIUS,
                type=ENEMY_BALL,
                image_type=ENEMY_BALL,
                expire_time=BALL_EXPIRE,
                active=aligned_x | aligned_y,
            )
            free = ~pool.active
            ball_slot = jnp.argmax(free)
            pool = jax.lax.cond(
                (aligned_x | aligned_y) & free.any(),
                lambda p: jax.tree.map(lambda buf, value: buf.at[ball_slot].set(value), p, thrown),
                lambda p: p,
                pool,
            )
            # a thrower squares up to the agent and cools down
            pool = pool._replace(
                vx=jnp.where(aligned_x, 0.0, jnp.where(aligned_y, toward_x * ENEMY_VEL, pool.vx)),
                vy=jnp.where(aligned_y, 0.0, jnp.where(aligned_x, toward_y * ENEMY_VEL, pool.vy)),
                fire_time=jnp.where(aligned_x | aligned_y, now + cool, pool.fire_time),
            )
            return pool, state, agent

        pool, state, agent = jax.lax.fori_loop(0, MAX_ENTITIES, enemy_turn, (pool, state, agent))
        pool = jax.vmap(engine.entity_step)(pool)

        # balls leave through the walls
        gone = ((pool.type == PLAYER_BALL) | (pool.type == ENEMY_BALL)) & (
            (pool.x < pool.rx) | (pool.x > WORLD_DIM - pool.rx) | (pool.y < pool.ry) | (pool.y > WORLD_DIM - pool.ry)
        )
        pool = pool._replace(active=pool.active & ~gone)

        # ---- collisions ------------------------------------------------------- #
        def collide(index, carry):
            pool, reward = carry
            src = jax.tree.map(lambda field: field[index], pool)
            is_src = pool.collides_with_entities[index] & pool.active[index]
            hit = engine.collides(src, pool) & pool.active & is_src
            hit = hit.at[index].set(False)

            ball = src.type == PLAYER_BALL
            lava_hit = (hit & (pool.type == LAVA_WALL)).any()
            enemy_hit = hit & (pool.type == ENEMY)
            pool = pool._replace(health=pool.health - jnp.where(ball, enemy_hit.astype(jnp.float32), 0.0))
            killed = ball & enemy_hit & (pool.health <= 0)
            pool = pool._replace(active=pool.active & ~killed)
            reward = reward + ENEMY_REWARD * killed.sum()
            erase_self = ball & (lava_hit | enemy_hit.any())
            pool = pool._replace(active=pool.active.at[index].set(pool.active[index] & ~erase_self))

            wall = src.type == LAVA_WALL
            struck = wall & hit & ((pool.type == PLAYER_BALL) | (pool.type == ENEMY_BALL))
            pool = pool._replace(active=pool.active & ~struck)
            return pool, reward

        pool, reward = jax.lax.fori_loop(0, MAX_ENTITIES, collide, (pool, jnp.zeros((), jnp.float32)))

        # ---- the door, the agent's contact and the counters ------------------- #
        num_enemies = ((pool.type == ENEMY) & pool.active).sum()
        pool = pool._replace(
            image_type=jnp.where(pool.type == DOOR, jnp.where(num_enemies == 0, DOOR_OPEN, DOOR), pool.image_type)
        )

        touching = engine.collides(agent, pool) & pool.active
        reached_door = (touching & (pool.type == DOOR) & (num_enemies == 0)).any()
        killed_by = (touching & ((pool.type == ENEMY) | (pool.type == ENEMY_BALL) | (pool.type == LAVA_WALL))).any()
        terminated = reached_door | killed_by | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)
        reward = reward + jnp.where(reached_door, COMPLETION_BONUS, 0.0)

        stepped = DodgeballState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            num_enemies=num_enemies,
            last_fire_time=last_fire_time,
            last_move_action=last_move_action,
            steps=env_state.steps + 1,
            key=key,
            rng=state,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: DodgeballState,
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

    def observe(self, env_state: DodgeballState) -> jax.Array:
        """The whole arena: ``center_agent`` is off."""
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
            background=FLOOR_COLOUR,
        )
