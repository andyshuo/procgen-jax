"""A pure-JAX port of ProcGen's ``ninja``, from ``procgen/src/games/ninja.cpp``.

A side-scrolling platformer: the level wanders to the right across platforms with a
floor of fire and bombs on the ceilings, the zombie charges a jump while the action
is held, and action 9-12 throw stars up, diagonally or straight ahead.

Deviations: the jump-charge bar is a HUD element and is not drawn, the explosions a
star sets off when it hits a bomb are visual and not spawned, and the sprite does not
rotate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``ninja.cpp`` object ids.
GOAL = 1
BOMB_TILE = 6
THROWING_STAR = 7
FIRE_TILE = 14
WALL_MID = 20
PLAYER_JUMP = 9
PLAYER_RIGHT1 = 12
PLAYER_RIGHT2 = 13

GOAL_REWARD = 10.0
WORLD_DIM = 64
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.5
GRAVITY = 0.2
MAX_JUMP = 1.5
AIR_CONTROL = 0.15
TERMINAL_VELOCITY = -2.0
JUMP_CHARGE_INC = 0.25
VISIBILITY = 16
NUM_WALL_THEMES = 3
AGENT_RADIUS = 0.5
STAR_RADIUS = 0.25
STAR_SPEED = 1.0
STAR_EXPIRE = 15
FIRE_COOLDOWN = 3
#: ``randn(3) + 1``, and each section is at most two edges long.
MAX_DIFFICULTY = 3
MAX_SECTIONS = 2 * MAX_DIFFICULTY
MAX_EDGES = 2
MAX_ENTITIES = 48
MAX_SUB_STEPS = 6
#: The grid keeps ProcGen's own ``SPACE`` id, like the generators use.
SPACE = 100

WALL_SPRITES = ['tile_bricksGrey', 'tile_bricksGrown', 'tile_bricksRed']
GOAL_SPRITES = ['shroom1', 'shroom2', 'shroom3', 'shroom4', 'shroom5', 'shroom6']

TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32)
    .at[WALL_MID]
    .set(sprites.row('misc/ninja/tile_bricksGrey'))
    .at[BOMB_TILE]
    .set(sprites.row('misc/ninja/bomb'))
    .at[FIRE_TILE]
    .set(sprites.row('misc/ninja/bomb'))
)
WALL_ROWS = jnp.asarray([sprites.row(f'misc/ninja/{name}') for name in WALL_SPRITES], jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/ninja/zombie_idle'))
    .at[PLAYER_JUMP, 0]
    .set(sprites.row('misc/ninja/zombie_jump'))
    .at[PLAYER_RIGHT1, 0]
    .set(sprites.row('misc/ninja/zombie_walk1'))
    .at[PLAYER_RIGHT2, 0]
    .set(sprites.row('misc/ninja/zombie_walk2'))
    .at[THROWING_STAR, 0]
    .set(sprites.row('misc/ninja/saw'))
    .at[BOMB_TILE, 0]
    .set(sprites.row('misc/ninja/bomb'))
)
for _theme, _name in enumerate(GOAL_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[GOAL, _theme].set(sprites.row(f'misc/ninja/{_name}'))
#: The original paints a platformer background; this is a flat stand-in.
SKY_COLOUR = (60, 80, 110)


class NinjaState(NamedTuple):
    """Game state: the tower of platforms, the zombie and its stars."""

    grid: jax.Array  # (WORLD_DIM * WORLD_DIM,) WALL_MID, FIRE_TILE, BOMB_TILE or SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) the mushroom and the stars
    wall_theme: jax.Array
    has_support: jax.Array
    facing_right: jax.Array
    jump_charge: jax.Array
    last_fire_time: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def _fill(flat, x, y, dx, dy, value):
    """``fill_elem``: a rectangle of tiles, with traced bounds."""
    xs = jnp.arange(WORLD_DIM)
    ys = jnp.arange(WORLD_DIM)
    inside = (xs[None, :] >= x) & (xs[None, :] < x + dx) & (ys[:, None] >= y) & (ys[:, None] < y + dy)
    return jnp.where(inside.reshape(-1), value, flat)


class NinjaEnv(Env):
    """ProcGen's ``ninja`` as a functional, jit-friendly environment."""

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

        def reset(env_state: NinjaState) -> NinjaState:
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
    def reset_env_state(self, key: jax.Array) -> NinjaState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y
        state, difficulty = rng.randn(state, MAX_DIFFICULTY)
        difficulty = difficulty + 1
        state, wall_theme = rng.randn(state, NUM_WALL_THEMES)

        size = WORLD_DIM * WORLD_DIM
        flat = jnp.full((size,), SPACE, jnp.int32)
        # ``init_floor_and_walls``: a one-tile border
        flat = _fill(flat, 0, 0, WORLD_DIM, 1, WALL_MID)
        flat = _fill(flat, 0, 0, 1, WORLD_DIM, WALL_MID)
        flat = _fill(flat, WORLD_DIM - 1, 0, 1, WORLD_DIM, WALL_MID)
        flat = _fill(flat, 0, WORLD_DIM - 1, WORLD_DIM, 1, WALL_MID)

        state, flat, goal_x, goal_y = self._generate(state, difficulty, flat)

        state, goal_theme = rng.randn(state, len(GOAL_SPRITES))
        pool = empty_pool()
        goal = engine.entity(
            goal_x.astype(jnp.float32) + 0.5,
            goal_y.astype(jnp.float32) + 0.5,
            rx=0.5,
            ry=0.5,
            type=GOAL,
            image_type=GOAL,
            image_theme=goal_theme,
            active=True,
        )
        pool = jax.tree.map(lambda buf, value: buf.at[0].set(value), pool, goal)

        agent = engine.entity(
            1 + AGENT_RADIUS,
            WORLD_DIM / 2 + AGENT_RADIUS,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )
        return NinjaState(
            grid=flat,
            agent=agent,
            entities=pool,
            wall_theme=wall_theme,
            has_support=jnp.asarray(False),
            facing_right=jnp.asarray(True),
            jump_charge=jnp.zeros((), jnp.float32),
            last_fire_time=jnp.zeros((), jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    def _generate(self, state, difficulty, flat):
        """``generate_coin_to_the_right``: the wandering platforms and the mushroom."""
        min_gap = difficulty - 1
        min_plat_w = 1
        inc_dy = 4
        bomb_prob = 0.25 * (difficulty - 1).astype(jnp.float32)
        max_gap_inc = jnp.where(difficulty == 1, 1, 2)
        max_dy = jnp.floor(MAX_JUMP * MAX_JUMP / (2 * GRAVITY) - 0.5).astype(jnp.int32)

        state, sections_draw = rng.randn(state, difficulty)
        num_sections = sections_draw + difficulty

        start_x = jnp.asarray(5, jnp.int32)
        curr_x = start_x
        curr_y = jnp.asarray(WORLD_DIM // 2, jnp.int32)
        min_y = curr_y
        ceil_step = 1 + difficulty // 3

        flat = _fill(flat, 0, 0, start_x, curr_y, WALL_MID)
        flat = _fill(flat, 0, curr_y + 8, start_x, WORLD_DIM - curr_y - 8, WALL_MID)

        for section in range(MAX_SECTIONS):
            live = section < num_sections
            prev_x = curr_x
            prev_y = curr_y
            state, edges_draw = jax.lax.cond(
                live, lambda st: rng.randn(st, 2), lambda st: (st, jnp.zeros((), jnp.int32)), state
            )
            num_edges = edges_draw + 1
            max_y = jnp.asarray(-1, jnp.int32)
            last_edge_y = jnp.asarray(-1, jnp.int32)

            for edge in range(MAX_EDGES):
                active = live & (edge < num_edges)
                curr_x = prev_x + edge
                stop = active & (curr_x + 15 < WORLD_DIM)
                curr_y = prev_y

                state, dy_draw = jax.lax.cond(
                    stop, lambda st: rng.randn(st, inc_dy), lambda st: (st, jnp.zeros((), jnp.int32)), state
                )
                dy = dy_draw + 1 + ceil_step
                dy = jnp.minimum(dy, max_dy)

                state, flip_draw = jax.lax.cond(
                    stop & (curr_y >= 5) & (curr_y < WORLD_DIM - 15),
                    lambda st: rng.rand01(st),
                    lambda st: (st, jnp.ones((), jnp.float32)),
                    state,
                )
                dy = jnp.where((curr_y >= WORLD_DIM - 15) | ((curr_y >= 5) & (flip_draw < 0.4)), -dy, dy)
                curr_y = jnp.maximum(curr_y + dy, 3)
                curr_y = jnp.where(jnp.abs(curr_y - last_edge_y) <= 1, last_edge_y + 2, curr_y)

                state, dx_draw = jax.lax.cond(
                    stop, lambda st: rng.randn(st, 3), lambda st: (st, jnp.zeros((), jnp.int32)), state
                )
                dx = min_plat_w + dx_draw
                flat = jnp.where(stop, _fill(flat, curr_x, curr_y - 1, dx, 1, WALL_MID), flat)
                curr_x = curr_x + dx

                state, gap_draw = jax.lax.cond(
                    stop,
                    lambda st: rng.randn(st, max_gap_inc + 1),
                    lambda st: (st, jnp.zeros((), jnp.int32)),
                    state,
                )
                curr_x = curr_x + min_gap + gap_draw

                max_y = jnp.where(stop, jnp.maximum(max_y, curr_y), max_y)
                min_y = jnp.where(stop, jnp.minimum(min_y, curr_y), min_y)
                last_edge_y = jnp.where(stop, curr_y, last_edge_y)

            state, bomb_draw = jax.lax.cond(
                live & (bomb_prob > 0),
                lambda st: rng.rand01(st),
                lambda st: (st, jnp.ones((), jnp.float32)),
                state,
            )
            state, bomb_x = jax.lax.cond(
                live & (bomb_prob > 0) & (bomb_draw < bomb_prob),
                lambda st, span=curr_x - prev_x + 1: rng.randn(st, jnp.maximum(span, 1)),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            bomb = live & (bomb_prob > 0) & (bomb_draw < bomb_prob)
            flat = jnp.where(
                bomb,
                flat.at[jnp.clip((max_y + 2) * WORLD_DIM + bomb_x + prev_x, 0, size_guard())].set(BOMB_TILE),
                flat,
            )

            ceiling_height = 11
            ceiling_start = max_y - 1 + ceiling_height
            flat = jnp.where(
                live,
                _fill(flat, prev_x, ceiling_start, curr_x - prev_x, WORLD_DIM - ceiling_start, WALL_MID),
                flat,
            )

        flat = _fill(flat, curr_x, curr_y - 1, 1, 1, WALL_MID)
        flat = _fill(flat, curr_x, curr_y + 6, 1, WORLD_DIM - curr_y - 6, WALL_MID)

        fire_y = jnp.maximum(min_y - 2, 1)
        flat = _fill(flat, start_x, 0, WORLD_DIM - start_x, fire_y, WALL_MID)
        flat = _fill(flat, start_x, fire_y, WORLD_DIM - start_x, 1, FIRE_TILE)
        flat = _fill(flat, curr_x + 1, 0, WORLD_DIM - curr_x - 1, WORLD_DIM, WALL_MID)
        return state, flat, curr_x, curr_y

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: NinjaState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1
        grid = env_state.grid
        agent = env_state.agent

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)
        action_vx = move_action // 3 - 1
        action_vy = jnp.maximum(move_action % 3 - 1, 0)

        # ---- set_action_xy: footing and the jump charge ----------------------- #
        view = grid.reshape(WORLD_DIM, WORLD_DIM)
        below = []
        for offset in (AGENT_RADIUS - 0.01, -(AGENT_RADIUS - 0.01)):
            cell_x = jnp.clip(jnp.floor(agent.x + offset).astype(jnp.int32), 0, WORLD_DIM - 1)
            cell_y = jnp.clip(jnp.floor(agent.y - (AGENT_RADIUS + 0.01)).astype(jnp.int32), 0, WORLD_DIM - 1)
            below.append(view[cell_y, cell_x])
        support_tiles = jnp.stack(below)
        has_support = ((support_tiles == WALL_MID) | (support_tiles == FIRE_TILE)).any()
        charging = has_support & (action_vy == 1)
        action_vy = jnp.where(charging, action_vy, 0)
        jump_charge = jnp.where(
            charging, jnp.minimum(env_state.jump_charge + JUMP_CHARGE_INC, 1.0), env_state.jump_charge
        )
        jump_charge = jnp.where(has_support, jump_charge, 0.0)
        facing_right = jnp.where(action_vx > 0, True, jnp.where(action_vx < 0, False, env_state.facing_right))

        # ---- update_agent_velocity ------------------------------------------- #
        mixrate_x = jnp.where(has_support, MIXRATE, MIXRATE * AIR_CONTROL)
        vx = (1 - mixrate_x) * agent.vx + mixrate_x * MAX_SPEED * action_vx
        release = (action_vy < 1) & (jump_charge > 0)
        vy = jnp.where(release, jump_charge * MAX_JUMP, agent.vy)
        vy = jnp.where((~has_support) & (vy > TERMINAL_VELOCITY), vy - GRAVITY, vy)
        # the C++ only flips the facing when the action actually moves the zombie
        reflected = jnp.where(action_vx > 0, False, jnp.where(action_vx < 0, True, env_state.agent.is_reflected))
        agent = agent._replace(vx=vx, vy=vy, is_reflected=reflected)
        jump_charge = jnp.where(release, 0.0, jump_charge)

        agent, _ = engine.step_object(
            agent,
            view,
            out_of_bounds=jnp.asarray(WALL_MID),
            is_blocked=lambda tile, is_horizontal: tile == WALL_MID,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- throwing stars --------------------------------------------------- #
        firing = (special_action > 0) & ((now - env_state.last_fire_time) >= FIRE_COOLDOWN)
        theta = jnp.select(
            [special_action == 1, special_action == 2, special_action == 3],
            [0.0, jnp.pi / 4, jnp.pi / 2],
            -jnp.pi / 4,
        )
        theta = jnp.where(agent.is_reflected, jnp.pi - theta, theta)
        star = engine.entity(
            agent.x,
            agent.y,
            STAR_SPEED * jnp.cos(theta),
            STAR_SPEED * jnp.sin(theta),
            rx=STAR_RADIUS,
            ry=STAR_RADIUS,
            type=THROWING_STAR,
            image_type=THROWING_STAR,
            collides_with_entities=True,
            expire_time=STAR_EXPIRE,
            smart_step=True,
            active=firing,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            firing & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, star),
            lambda p: p,
            pool,
        )
        last_fire_time = jnp.where(firing, now, env_state.last_fire_time)

        # ---- the stars hit bricks and bombs ----------------------------------- #
        is_star = pool.active & (pool.type == THROWING_STAR)
        corners = []
        for i in (0, 1):
            for j in (0, 1):
                cx = jnp.clip((pool.x + pool.rx * (2 * i - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                cy = jnp.clip((pool.y + pool.ry * (2 * j - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                corners.append(view[cy, cx])
        corner_tiles = jnp.stack(corners, axis=1)
        hit_wall = (corner_tiles == WALL_MID).any(axis=1) & is_star
        hit_bomb = (corner_tiles == BOMB_TILE).any(axis=1) & is_star
        pool = pool._replace(active=pool.active & ~(hit_wall | hit_bomb))
        # a star that finds a bomb clears the bomb
        bomb_cells = jnp.zeros((size_guard(),), bool)
        for i in (0, 1):
            for j in (0, 1):
                cx = jnp.clip((pool.x + pool.rx * (2 * i - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                cy = jnp.clip((pool.y + pool.ry * (2 * j - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                bomb_cells = bomb_cells.at[(cy * WORLD_DIM + cx).reshape(-1)].set(
                    (corner_tiles[:, 2 * i + j] == BOMB_TILE) & hit_bomb
                )
        grid = jnp.where(bomb_cells[: grid.shape[0]], SPACE, grid)

        # ---- contact with the zombie ------------------------------------------ #
        touching = engine.collides(agent, pool) & pool.active
        reached_goal = (touching & (pool.type == GOAL)).any()
        reward = jnp.where(reached_goal, GOAL_REWARD, 0.0)

        # ---- what the zombie is standing in ----------------------------------- #
        feet = jnp.zeros((size_guard(),), bool)
        for i in (0, 1):
            for j in (0, 1):
                cx = jnp.clip((agent.x + agent.rx * (2 * i - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                cy = jnp.clip((agent.y + agent.ry * (2 * j - 1)).astype(jnp.int32), 0, WORLD_DIM - 1)
                feet = feet.at[(cy * WORLD_DIM + cx).reshape(-1)].set(True)
        standing_in = jnp.where(feet[: grid.shape[0]], grid, SPACE)
        burned = ((standing_in == FIRE_TILE) | (standing_in == BOMB_TILE)).any()
        terminated = reached_goal | burned

        image_type = jnp.where(
            (jnp.abs(agent.vx) < 0.01) & (action_vx == 0) & has_support,
            engine.PLAYER,
            jnp.where((now // 5 % 2 == 0) | ~has_support, PLAYER_RIGHT1, PLAYER_RIGHT2),
        )
        stepped = NinjaState(
            grid=grid,
            agent=agent._replace(image_type=image_type),
            entities=pool,
            wall_theme=env_state.wall_theme,
            has_support=has_support,
            facing_right=facing_right,
            jump_charge=jump_charge,
            last_fire_time=last_fire_time,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: NinjaState,
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

    def observe(self, env_state: NinjaState) -> jax.Array:
        """``visibility = 16`` around the zombie, with the level's brick theme."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        tile_rows = TILE_ROWS.at[WALL_MID].set(WALL_ROWS[env_state.wall_theme])
        return render_textured(
            env_state.grid.reshape(WORLD_DIM, WORLD_DIM),
            jnp.stack([env_state.agent.x, env_state.agent.y]),
            bodies,
            tile_rows,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            VISIBILITY,
            jnp.asarray(WALL_MID),
            background=SKY_COLOUR,
        )


def size_guard() -> int:
    """The flat grid size, used for the sink slots of the scatter helpers."""
    return WORLD_DIM * WORLD_DIM + 1
