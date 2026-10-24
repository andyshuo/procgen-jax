"""A pure-JAX port of ProcGen's ``fruitbot``, from ``procgen/src/games/fruitbot.cpp``.

A top-down shaft: the robot drifts downward (the action's vertical component is a
constant), left and right are yours, and ten barrier rows with gaps have to be
threaded.  Fruit pays +1, junk costs -4, a barrier or a locked door ends the run,
and a present at the bottom pays +10.  Action 9 shoots a key that opens locks.

Deviations: junk never moves, so the original's "junk bounces off barriers" rule
never fires; the top-down background is flat.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``fruitbot.cpp`` object ids.
BARRIER = 1
OUT_OF_BOUNDS_WALL = 2
PLAYER_BULLET = 3
BAD_OBJ = 4
GOOD_OBJ = 7
LOCKED_DOOR = 10
LOCK = 11
PRESENT = 12

COMPLETION_BONUS = 10.0
POSITIVE_REWARD = 1.0
PENALTY = -4.0
KEY_DURATION = 8
DOOR_ASPECT_RATIO = 3.25

WORLD_WIDTH = 20
WORLD_HEIGHT = 60
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.85
#: ``set_action_xy`` pins the vertical action: the robot always sinks.
ACTION_VY = 0.2
AGENT_RADIUS = 0.5
BULLET_RADIUS = 0.25
BULLET_SPEED = 0.5
#: ``game_reset``'s level shape.
NUM_WALLS = 10
MIN_SEP = 4
BUF_H = 4
MIN_PCT = 0.1
DOOR_PROB = 0.125
WALL_RY = 0.3
LOCK_RX = 0.25
LOCK_RY = 0.45
GROUP_SIZE = 6
NUM_PRESENTS = WORLD_WIDTH
#: Ten rows of two barriers, ten doors and locks, twenty presents, and up to 38
#: pieces of fruit and junk.
MAX_ENTITIES = 128
MAX_SUB_STEPS = 6
REPOSITION_ATTEMPTS = 100
#: ``randn(10) + 10``, so at most nineteen of each kind.
MAX_SPAWNS = 19

GOOD_SPRITES = [f'fruit{index}' for index in range(1, 7)]
BAD_SPRITES = [f'food{index}' for index in range(1, 7)]
PRESENT_SPRITES = [f'present{index}' for index in range(1, 4)]

TILE_SPRITES = jnp.full((64,), sprites.row('_transparent'), jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/fruitbot/robot_3Dblue'))
    .at[BARRIER, 0]
    .set(sprites.row('misc/fruitbot/tileStone_slope'))
    .at[OUT_OF_BOUNDS_WALL, 0]
    .set(sprites.row('misc/fruitbot/tileStone_slope'))
    .at[PLAYER_BULLET, 0]
    .set(sprites.row('misc/fruitbot/keyRed2'))
    .at[LOCKED_DOOR, 0]
    .set(sprites.row('misc/fruitbot/fenceYellow'))
    .at[LOCK, 0]
    .set(sprites.row('misc/fruitbot/lockRed2'))
)
for _theme, _name in enumerate(GOOD_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[GOOD_OBJ, _theme].set(sprites.row(f'misc/fruitbot/{_name}'))
for _theme, _name in enumerate(BAD_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[BAD_OBJ, _theme].set(sprites.row(f'misc/fruitbot/{_name}'))
for _theme, _name in enumerate(PRESENT_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[PRESENT, _theme].set(sprites.row(f'misc/fruitbot/{_name}'))

GOOD_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/fruitbot/{name}')] for name in GOOD_SPRITES])
BAD_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/fruitbot/{name}')] for name in BAD_SPRITES])
#: The original draws a top-down floor; this is a flat stand-in.
FLOOR_COLOUR = (56, 48, 42)


class FruitBotState(NamedTuple):
    """Game state: the shaft, the robot, the fruit and the barriers."""

    grid: jax.Array  # (WORLD_HEIGHT, WORLD_WIDTH), all SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) barriers, locks, fruit and presents
    last_fire_time: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class FruitBotEnv(Env):
    """ProcGen's ``fruitbot`` as a functional, jit-friendly environment."""

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

        def reset(env_state: FruitBotState) -> FruitBotState:
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
        draws = (high - low) > 2 * radius
        state, draw = jax.lax.cond(
            draws & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        value = jnp.where(draws, (high - low - 2 * radius) * draw + radius + low, (high + low) / 2)
        return state, value

    def _reposition(self, state, pool, agent, radius, low_x, low_y, size_x, size_y, active):
        """``reposition``: retry inside the box until the body is clear."""
        start = (
            state,
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.asarray(False),
        )

        def attempt(_, carry):
            state, x, y, done = carry
            state, nx = self._rand_pos(state, radius, low_x, low_x + size_x, active & ~done)
            state, ny = self._rand_pos(state, radius, low_y, low_y + size_y, active & ~done)
            candidate = engine.entity(nx, ny, rx=radius, ry=radius)
            blocked = (engine.collides(candidate, pool) & pool.active).any() | engine.collides(candidate, agent)
            fresh = active & ~done & ~blocked
            return state, jnp.where(fresh, nx, x), jnp.where(fresh, ny, y), done | fresh

        return jax.lax.fori_loop(0, REPOSITION_ATTEMPTS, attempt, start)

    def reset_env_state(self, key: jax.Array) -> FruitBotState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, start_x = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        agent = engine.entity(
            start_x * (WORLD_WIDTH - 2 * AGENT_RADIUS) + AGENT_RADIUS,
            AGENT_RADIUS,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )

        pool = empty_pool()
        count = jnp.zeros((), jnp.int32)

        def write(pool, count, ent, active):
            slot = jnp.clip(count, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=slot, on=active: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                ent,
            )
            return pool, count + active

        # ``partition(height - min_sep * num_walls - buf_h, num_walls)``: one draw per
        # unit, dropping it into a random bin.
        bins = jnp.zeros((NUM_WALLS,), jnp.int32)
        for _ in range(WORLD_HEIGHT - MIN_SEP * NUM_WALLS - BUF_H):
            state, bin_draw = rng.randn(state, NUM_WALLS)
            bins = bins.at[bin_draw].add(1)

        height = jnp.zeros((), jnp.float32)
        for wall in range(NUM_WALLS):
            dy = MIN_SEP + bins[wall]
            height = height + dy.astype(jnp.float32)
            state, door_draw = jax.lax.cond(
                dy > 5,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.ones((), jnp.float32)),
                state,
            )
            use_door = (dy > 5) & (door_draw < DOOR_PROB)

            state, pct_draw = rng.rand01(state)
            pct = MIN_PCT + 0.2 * pct_draw
            lock_pct_w = 2 * LOCK_RX / WORLD_WIDTH
            door_pct_w = (WALL_RY * 2 * DOOR_ASPECT_RATIO) / WORLD_WIDTH
            door_pct = 2 * lock_pct_w + door_pct_w * jnp.ceil((pct + 0.1 - 2 * lock_pct_w) / door_pct_w)
            pct = jnp.where(use_door, door_pct, pct)

            gap_w = pct * WORLD_WIDTH
            state, w1_draw = rng.rand01(state)
            w1 = w1_draw * (WORLD_WIDTH - gap_w)
            w2 = WORLD_WIDTH - w1 - gap_w

            left = engine.entity(
                w1 / 2,
                height,
                rx=w1 / 2,
                ry=WALL_RY,
                type=BARRIER,
                image_type=BARRIER,
            )
            pool, count = write(pool, count, left, jnp.asarray(True))
            right = engine.entity(
                WORLD_WIDTH - w2 / 2,
                height,
                rx=w2 / 2,
                ry=WALL_RY,
                type=BARRIER,
                image_type=BARRIER,
            )
            pool, count = write(pool, count, right, jnp.asarray(True))

            state, side_draw = jax.lax.cond(
                use_door,
                lambda s: rng.randn(s, 2),
                lambda s: (s, jnp.zeros((), jnp.int32)),
                state,
            )
            lock_x = w1 + LOCK_RX + side_draw * (gap_w - 2 * LOCK_RX)
            door_x = w1 + gap_w / 2 - (side_draw * 2 - 1) * LOCK_RX
            door = engine.entity(
                door_x,
                height,
                rx=gap_w / 2 - LOCK_RX,
                ry=WALL_RY,
                type=LOCKED_DOOR,
                image_type=LOCKED_DOOR,
            )
            pool, count = write(pool, count, door, use_door)
            lock = engine.entity(
                lock_x,
                height - LOCK_RY + WALL_RY,
                rx=LOCK_RX,
                ry=LOCK_RY,
                type=LOCK,
                image_type=LOCK,
            )
            pool, count = write(pool, count, lock, use_door)

        # ---- the presents along the bottom ------------------------------------ #
        for index in range(NUM_PRESENTS):
            state, theme = rng.randn(state, len(PRESENT_SPRITES))
            present = engine.entity(
                index + 0.5,
                WORLD_HEIGHT - 0.5,
                rx=0.5,
                ry=0.5,
                type=PRESENT,
                image_type=PRESENT,
                image_theme=theme,
            )
            pool, count = write(pool, count, present, jnp.asarray(True))

        # ---- the fruit and the junk ------------------------------------------- #
        state, good_draw = rng.randn(state, 10)
        state, bad_draw = rng.randn(state, 10)
        num_good = good_draw + 10
        num_bad = bad_draw + 10
        first_piece = count

        def spawn_piece(slot, carry):
            state, pool, count = carry
            is_good = slot < num_good
            is_bad = (slot >= num_good) & (slot < num_good + num_bad)
            spawn = is_good | is_bad
            state, x, y, _ = self._reposition(state, pool, agent, 0.5, 0.0, 0.0, WORLD_WIDTH, WORLD_HEIGHT, spawn)
            piece = engine.entity(
                x,
                y,
                rx=0.5,
                ry=0.5,
                type=jnp.where(is_good, GOOD_OBJ, BAD_OBJ),
                image_type=jnp.where(is_good, GOOD_OBJ, BAD_OBJ),
                active=spawn,
            )
            pool, count = write(pool, count, piece, spawn)
            return state, pool, count

        state, pool, count = jax.lax.fori_loop(0, 2 * MAX_SPAWNS, spawn_piece, (state, pool, count))

        # ---- themes and aspect ratios for the fruit and the junk -------------- #
        goods = pool.type == GOOD_OBJ
        bads = pool.type == BAD_OBJ

        def dress(slot, carry):
            state, pool = carry
            index = first_piece + slot
            themed = (goods[index] | bads[index]) & pool.active[index]
            state, theme = jax.lax.cond(
                themed,
                lambda st: rng.randn(st, GROUP_SIZE),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            theme = jnp.where(themed, theme, pool.image_theme[index])
            aspect = jnp.where(
                goods[index],
                GOOD_ASPECTS[jnp.clip(theme, 0, len(GOOD_SPRITES) - 1)],
                BAD_ASPECTS[jnp.clip(theme, 0, len(BAD_SPRITES) - 1)],
            )
            rx = jnp.where(aspect > 1, pool.rx[index], pool.ry[index] * aspect)
            ry = jnp.where(aspect > 1, pool.rx[index] / aspect, pool.ry[index])
            return state, pool._replace(
                image_theme=pool.image_theme.at[index].set(theme),
                rx=pool.rx.at[index].set(jnp.where(themed, rx, pool.rx[index])),
                ry=pool.ry.at[index].set(jnp.where(themed, ry, pool.ry[index])),
            )

        state, pool = jax.lax.fori_loop(0, 2 * MAX_SPAWNS, dress, (state, pool))

        return FruitBotState(
            grid=jnp.full((WORLD_HEIGHT, WORLD_WIDTH), SPACE, jnp.int32),
            agent=agent,
            entities=pool,
            last_fire_time=jnp.zeros((), jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: FruitBotState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)
        action_vx = move_action // 3 - 1

        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * ACTION_VY
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy, is_reflected=action_vx < 0)

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(OUT_OF_BOUNDS_WALL),
            is_blocked=lambda tile, is_horizontal: tile == OUT_OF_BOUNDS_WALL,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=jnp.asarray(ACTION_VY),
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- fire a key ------------------------------------------------------- #
        can_fire = (special_action == 1) & ((now - env_state.last_fire_time) >= KEY_DURATION)
        shot = engine.entity(
            agent.x,
            agent.y,
            0.0,
            BULLET_SPEED,
            rx=BULLET_RADIUS,
            ry=BULLET_RADIUS,
            type=PLAYER_BULLET,
            image_type=PLAYER_BULLET,
            collides_with_entities=True,
            expire_time=KEY_DURATION,
            active=can_fire,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            can_fire & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, shot),
            lambda p: p,
            pool,
        )
        last_fire_time = jnp.where(can_fire, now, env_state.last_fire_time)

        # ---- the key opens a lock and its door -------------------------------- #
        is_key = pool.collides_with_entities & pool.active & (pool.type == PLAYER_BULLET)
        shot_index = jnp.argmax(is_key)
        shot = jax.tree.map(lambda field: field[shot_index], pool)
        struck = engine.collides(shot, pool) & pool.active & is_key.any()
        struck = struck.at[shot_index].set(False)
        hit_barrier = (struck & (pool.type == BARRIER)).any()
        hit_lock = struck & (pool.type == LOCK)
        opened = hit_lock.any()
        lock_index = jnp.argmax(hit_lock)
        door_y = pool.y[lock_index]
        same_height = (pool.type == LOCKED_DOOR) & (jnp.abs(pool.y - door_y) < 1.0)
        pool = pool._replace(active=pool.active & ~hit_lock)
        pool = pool._replace(active=pool.active & ~(opened & same_height))
        pool = pool._replace(active=pool.active.at[shot_index].set(pool.active[shot_index] & ~(hit_barrier | opened)))
        reward = jnp.zeros((), jnp.float32)

        # ---- contact with the agent ------------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        ate_good = touching & (pool.type == GOOD_OBJ)
        ate_bad = touching & (pool.type == BAD_OBJ)
        hit_barrier = touching & (pool.type == BARRIER)
        hit_door = touching & (pool.type == LOCKED_DOOR)
        got_present = touching & (pool.type == PRESENT)

        reward = reward + POSITIVE_REWARD * ate_good.sum() + PENALTY * ate_bad.sum()
        reward = reward + COMPLETION_BONUS * got_present.any()
        pool = pool._replace(active=pool.active & ~(ate_good | ate_bad))

        terminated = hit_barrier.any() | hit_door.any() | got_present.any()
        terminated = terminated | engine.is_out_of_bounds(agent, WORLD_WIDTH, WORLD_HEIGHT)

        stepped = FruitBotState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            last_fire_time=last_fire_time,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: FruitBotState,
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

    def observe(self, env_state: FruitBotState) -> jax.Array:
        """``choose_center``: the shaft is twenty tiles across and follows the robot."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        centre = jnp.asarray(
            [
                WORLD_WIDTH / 2,
                env_state.agent.y + WORLD_WIDTH / 2 - 2 * env_state.agent.ry,
            ]
        )
        return render_textured(
            env_state.grid,
            centre,
            bodies,
            TILE_SPRITES,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            WORLD_WIDTH,
            jnp.asarray(SPACE),
            background=FLOOR_COLOUR,
        )
