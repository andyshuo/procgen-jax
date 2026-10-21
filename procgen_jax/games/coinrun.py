"""A pure-JAX port of ProcGen's ``coinrun``, from ``procgen/src/games/coinrun.cpp``.

Level layout is generated like the original: the floor is carved from left to
right in "sections", each rising or dropping by a few tiles, and some sections
become pits filled with lava, saws or walking enemies.  Flat runs get saws,
enemies and crate piles; ``ENEMY_BARRIER`` tiles fence walking enemies into their
own run; the level ends with the coin (``GOAL``) on a one-tile plinth.

Physics is ``basic-abstract-game.cpp``'s, transcribed in :mod:`procgen_jax.engine`:
gravity is applied before movement, jumping is a velocity impulse clamped to
``max_jump``, and movement is sub-stepped so a fast body cannot tunnel through a
tile.

Deviations from the original, all deliberate:

* rendering uses flat colours instead of Kenney sprites;
* the 3x3 action grid is decoded as in the original, so action slot 5 (the "down"
  row) is the jump button -- that is ProcGen's encoding, not a bug;
* action slots 9-14 are no-ops, as in :mod:`procgen_jax.games.maze`;
* the observation is always agent-centred;
* entity pushing is not ported (``sub_step``'s ``push_obj`` recursion), and
  trails are not rendered;
* ``EasyMode`` is not implemented: every level uses the harder distribution.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_scene

# --------------------------------------------------------------------------- #
# object ids, copied from ``coinrun.cpp`` / ``object-ids.h``
# --------------------------------------------------------------------------- #
GOAL = 1
SAW = 2
SAW2 = 3
ENEMY = 5
ENEMY1 = 6
ENEMY2 = 7
PLAYER_JUMP = 9
PLAYER_RIGHT1 = 12
PLAYER_RIGHT2 = 13
WALL_MID = 15
WALL_TOP = 16
LAVA_MID = 17
LAVA_TOP = 18
ENEMY_BARRIER = 19
CRATE = 20

#: The engine's out-of-grid object id (``object-ids.h``).
WALL_OBJ = engine.WALL_OBJ

#: ``coinrun.cpp``: ``GOAL_REWARD = 10.0f``.
GOAL_REWARD = 10.0
#: ``main_width`` / ``main_height``.
WORLD_DIM = 64
#: ``visibility = 13`` in the ``CoinRun`` constructor.
VISIBILITY = 13
#: ProcGen exposes 15 discrete actions for every game.
NUM_ACTIONS = 15
#: ProcGen's ``timeout``.
TIMEOUT = 500

# Engine constants from ``CoinRun::game_reset``.
GRAVITY = 0.2
MAX_JUMP = 1.5
MAX_SPEED = 0.5
MIXRATE = 0.2
AIR_CONTROL = 0.15
AGENT_RX = 0.5
AGENT_RY = 0.5787
START_X = 1.5
START_Y = 1.5787

# Bounds that turn the original's data-dependent loops into static ones.
MAX_SECTIONS = 5  # num_sections = randn(dif) + dif, dif <= 3
MAX_DX = 9  # dx = randn(2*dif) + 3 + dif//3
MAX_PIT = 7  # pit_width <= max_dx
MAX_ENTITIES = 36  # worst case: every section a pit full of enemies
MAX_CRATE_PILE = 3
#: ``int(4*|v|)`` is at most 6 because ``|vx| <= maxspeed`` and ``|vy| <= max_jump``.
MAX_SUB_STEPS = 6


class CoinRunState(NamedTuple):
    """Game state: the tile grid, the agent, and two pools of level entities."""

    grid: jax.Array  # (64, 64) int32 tile ids
    agent: engine.Entity
    enemies: engine.Entity  # walking enemies, batched over MAX_ENTITIES
    statics: engine.Entity  # saws and crates, batched over MAX_ENTITIES
    num_enemies: jax.Array
    num_statics: jax.Array
    steps: jax.Array
    key: jax.Array
    last_agent_y: jax.Array
    has_support: jax.Array
    is_on_crate: jax.Array
    facing_right: jax.Array


# --------------------------------------------------------------------------- #
# level generation
# --------------------------------------------------------------------------- #
def empty_entity_pool(count: int) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    # ``entity`` leaves scalar fields scalar; a pool needs every field batched.
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def fill(grid: jax.Array, x, y, dx, dy, value) -> jax.Array:
    """``fill_elem``: paint a rectangle, clipped to the grid."""
    height, width = grid.shape
    xs = jnp.arange(width)[None, :]
    ys = jnp.arange(height)[:, None]
    mask = (xs >= x) & (xs < x + dx) & (ys >= y) & (ys < y + dy)
    return jnp.where(mask, jnp.asarray(value, jnp.int32), grid)


def _fill_block_top(grid, x, y, dx, dy, body_type, top_type):
    """``fill_block_top``: a solid body with a distinct top row."""
    grid = fill(grid, x, y, dx, dy - 1, body_type)
    return fill(grid, x, y + dy - 1, dx, 1, top_type)


def _fill_ground(grid, x, y, dx, dy):
    return _fill_block_top(grid, x, y, dx, dy, WALL_MID, WALL_TOP)


def _fill_lava(grid, x, y, dx, dy):
    return _fill_block_top(grid, x, y, dx, dy, LAVA_MID, LAVA_TOP)


def _draw(u, index, n):
    """``rand_gen.randn(n)``: an integer in ``[0, n)``, drawn from a uniform."""
    return jnp.clip(jnp.floor(u[..., index] * n).astype(jnp.int32), 0, jnp.maximum(n - 1, 0))


def _append(pool: engine.Entity, count, new: engine.Entity, active):
    """Write ``new`` at ``count`` and advance only when ``active``.

    A masked-off write still lands at ``count``, which is the slot the next active
    write would claim, so it can never clobber a live entity.
    """
    new = new._replace(active=active)
    slot = jnp.clip(count, 0, pool.x.shape[0] - 1)
    pool = jax.tree.map(lambda buf, val: buf.at[slot].set(val), pool, new)
    return pool, count + active.astype(jnp.int32)


def _push_saw(pool, count, x, y, active):
    return _append(pool, count, engine.entity(x + 0.5, y + 0.5, type=SAW), active)


def _push_enemy(pool, count, x, y, vx, active):
    ent = engine.entity(x + 0.5, y + 0.5, vx, 0.0, type=ENEMY, render_z=1, smart_step=True)
    return _append(pool, count, ent, active)


def _push_crate(pool, count, x, y, active):
    return _append(pool, count, engine.entity(x + 0.5, y + 0.5, type=CRATE), active)


def generate_level(key: jax.Array):
    """``generate_coin_to_the_right``.

    Returns ``(grid, enemies, statics, num_enemies, num_statics)``.
    """
    head_key, section_key = jax.random.split(key)
    head = jax.random.uniform(head_key, (4,), dtype=jnp.float32)
    section_u = jax.random.uniform(section_key, (MAX_SECTIONS, 20), dtype=jnp.float32)

    difficulty = _draw(head, 0, 3) + 1
    num_sections = _draw(head, 1, difficulty) + difficulty
    danger_type = _draw(head, 2, 3)

    # ``_max_dy``/``_max_dx`` come from the jump arc, as in the original.
    max_dy = jnp.asarray(int(MAX_JUMP * MAX_JUMP / (2 * GRAVITY) - 0.5), jnp.int32)
    max_dx = jnp.asarray(int(MAX_SPEED * 2 * MAX_JUMP / GRAVITY - 0.5), jnp.int32)

    grid = jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32)
    # ``init_floor_and_walls`` runs before the level is carved.
    grid = fill(grid, 0, 0, WORLD_DIM, 1, WALL_TOP)
    grid = fill(grid, 0, 0, 1, WORLD_DIM, WALL_MID)
    grid = fill(grid, WORLD_DIM - 1, 0, 1, WORLD_DIM, WALL_MID)
    grid = fill(grid, 0, WORLD_DIM - 1, WORLD_DIM, 1, WALL_MID)

    enemies = empty_entity_pool(MAX_ENTITIES)
    statics = empty_entity_pool(MAX_ENTITIES)

    def body(carry, section):
        grid, cx, cy, enemies, n_enemies, statics, n_statics = carry
        u = section_u[section]
        int_dif = difficulty // 3
        alive = (section < num_sections) & (cx + 15 < WORLD_DIM)

        # ---- where this section goes -------------------------------------- #
        dy = _draw(u, 0, 4) + 1 + int_dif
        dy = jnp.minimum(dy, max_dy)
        flip_high = cy >= 20  # too high: always go down
        flip_chance = (cy >= 5) & (_draw(u, 1, 2) == 1)
        dy = jnp.where(flip_high | flip_chance, -dy, dy)
        dx = _draw(u, 2, 2 * difficulty) + 3 + int_dif

        cy = jnp.where(alive, jnp.maximum(cy + dy, 1), cy)
        use_pit = alive & (dx > 7) & (cy > 3) & (_draw(u, 3, 20) >= difficulty)

        # ---- flat run: ground, a saw, an enemy and crate piles ------------ #
        flat = _fill_ground(grid, cx, 0, dx, cy)
        saw_x = cx + _draw(u, 10, dx - 2) + 1
        saw_on = (_draw(u, 11, 10) < 2 * difficulty) & (dx > 3)
        enemy_x = cx + _draw(u, 12, dx - 2) + 1
        enemy_on = (_draw(u, 13, 10) < difficulty) & (dx > 3) & (max_dx >= 4)

        # ---- pit: lava, or a row of hazards, sometimes bridged ------------- #
        x1 = _draw(u, 4, 3) + 1
        x2 = _draw(u, 5, 3) + 1
        too_wide = (dx - x1 - x2) > max_dx
        pit_width = jnp.where(too_wide, max_dx, dx - x1 - x2)
        x2 = jnp.where(too_wide, dx - x1 - pit_width, x2)

        pit = _fill_ground(grid, cx, 0, x1, cy)
        pit = _fill_ground(pit, cx + dx - x2, 0, x2, cy)
        lava_height = _draw(u, 6, cy - 3) + 1
        pit = jnp.where(
            use_pit & (danger_type == 0),
            _fill_lava(pit, cx + x1, 1, pit_width, lava_height),
            pit,
        )

        # Bridge geometry: narrow pits place a 1-2 tile shelf, wide ones a span.
        x3 = _draw(u, 7, 2) + jnp.where(pit_width == 5, 1, 2)
        x4 = _draw(u, 9, 2) + 2
        w1_narrow = _draw(u, 8, 2) + 1
        # Pits of 5 or 6 tiles use the narrow shelf; 7+ spans the pit with x3/x4.
        w1 = jnp.where(pit_width <= 6, w1_narrow, pit_width - x3 - x4)
        pit = jnp.where(
            use_pit & (pit_width > 4),
            _fill_ground(pit, cx + x1 + x3, cy - 1, w1, 1),
            pit,
        )

        grid = jnp.where(alive, jnp.where(use_pit, pit, flat), grid)

        # ---- entities ----------------------------------------------------- #
        enemy_vx = 0.15 * (2 * _draw(u, 14, 2) - 1)
        for step_index in range(MAX_PIT):
            in_pit = use_pit & (step_index < pit_width)
            statics, n_statics = _push_saw(
                statics,
                n_statics,
                cx + x1 + step_index,
                1,
                alive & in_pit & (danger_type == 1),
            )
            enemies, n_enemies = _push_enemy(
                enemies,
                n_enemies,
                cx + x1 + step_index,
                1,
                enemy_vx,
                alive & in_pit & (danger_type == 2),
            )

        on_flat = alive & ~use_pit
        statics, n_statics = _push_saw(statics, n_statics, saw_x, cy, on_flat & saw_on)
        enemies, n_enemies = _push_enemy(enemies, n_enemies, enemy_x, cy, enemy_vx, on_flat & enemy_on)

        for pile_index in range(2):
            crate_x = cx + _draw(u, 15 + 2 * pile_index, dx - 2) + 1
            crate_on = on_flat & (_draw(u, 16 + 2 * pile_index, 2) == 1)
            crate_on = crate_on & (crate_x != saw_x) & (crate_x != enemy_x)
            pile_height = _draw(u, 17 + 2 * pile_index, MAX_CRATE_PILE) + 1
            for level_index in range(MAX_CRATE_PILE):
                statics, n_statics = _push_crate(
                    statics,
                    n_statics,
                    crate_x,
                    cy + level_index,
                    crate_on & (level_index < pile_height),
                )

        # ---- fence the run so enemies stay in their section --------------- #
        previous = engine.tile_at(grid, cx - 1, cy, WALL_MID)
        grid = jnp.where(
            alive & ~((previous == WALL_MID) | (previous == WALL_TOP)),
            fill(grid, cx - 1, cy, 1, 1, ENEMY_BARRIER),
            grid,
        )
        cx = jnp.where(alive, cx + dx, cx)
        grid = jnp.where(alive, fill(grid, cx, cy, 1, 1, ENEMY_BARRIER), grid)

        return (grid, cx, cy, enemies, n_enemies, statics, n_statics), None

    carry = (
        grid,
        jnp.asarray(5, jnp.int32),
        jnp.asarray(1, jnp.int32),
        enemies,
        jnp.asarray(0, jnp.int32),
        statics,
        jnp.asarray(0, jnp.int32),
    )
    carry, _ = jax.lax.scan(body, carry, jnp.arange(MAX_SECTIONS))
    grid, cx, cy, enemies, n_enemies, statics, n_statics = carry

    # The coin sits on a plinth one tile below it; the rest of the world is wall.
    grid = fill(grid, cx, cy, 1, 1, GOAL)
    grid = _fill_ground(grid, cx, 0, 1, cy)
    grid = fill(grid, cx + 1, 0, WORLD_DIM - cx - 1, WORLD_DIM, WALL_MID)
    return grid, enemies, statics, n_enemies, n_statics


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class CoinRunEnv(Env):
    """ProcGen's ``coinrun`` as a functional, jit-friendly environment."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout

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

        def reset(env_state: CoinRunState) -> CoinRunState:
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

    # -- game rules -------------------------------------------------------- #
    def reset_env_state(self, key: jax.Array) -> CoinRunState:
        grid, enemies, statics, num_enemies, num_statics = generate_level(key)
        return CoinRunState(
            grid=grid,
            agent=engine.entity(
                START_X,
                START_Y,
                rx=AGENT_RX,
                ry=AGENT_RY,
                type=engine.PLAYER,
                render_z=1,
                smart_step=True,
            ),
            enemies=enemies,
            statics=statics,
            num_enemies=num_enemies,
            num_statics=num_statics,
            steps=jnp.zeros((), jnp.int32),
            key=key,
            last_agent_y=jnp.asarray(START_Y),
            has_support=jnp.asarray(True),
            is_on_crate=jnp.asarray(False),
            facing_right=jnp.asarray(True),
        )

    def step_env_state(self, env_state: CoinRunState, action: jax.Array):
        grid = env_state.grid
        agent = env_state.agent
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)

        # ---- game_step: decode the action ---------------------------------- #
        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        # ---- set_action_xy: is the agent standing on something? ------------- #
        below = jnp.stack(
            [
                engine.tile_at(grid, agent.x + s * (AGENT_RX - 0.01), agent.y - (AGENT_RY + 0.01), WALL_MID)
                for s in (-1, 1)
            ]
        )
        can_support = (below == WALL_MID) | (below == WALL_TOP)
        has_support = (env_state.is_on_crate | can_support.any()) & (agent.vy == 0)
        action_vy = jnp.where(action_vy == 1, jnp.where(has_support, action_vy, 0), action_vy)
        facing_right = jnp.where(action_vx > 0, True, jnp.where(action_vx < 0, False, env_state.facing_right))

        # ---- update_agent_velocity (coinrun's override) --------------------- #
        mixrate_x = jnp.where(has_support, MIXRATE, MIXRATE * AIR_CONTROL)
        vx = (1 - mixrate_x) * agent.vx + mixrate_x * MAX_SPEED * action_vx
        vx = jnp.where(jnp.abs(vx) < mixrate_x * MAX_SPEED, 0.0, vx)
        vy = jnp.where(action_vy > 0, MAX_JUMP, jnp.where(has_support, agent.vy + 0.2 * action_vy, agent.vy))
        vy = jnp.where(~(has_support & (action_vy > 0)), engine.clamp_abs(vy - GRAVITY, MAX_JUMP), vy)
        agent = agent._replace(vx=vx, vy=vy)

        # ---- step_entities: enemies move before the agent ------------------- #
        enemy_slots = jnp.arange(MAX_ENTITIES) < env_state.num_enemies
        static_slots = jnp.arange(MAX_ENTITIES) < env_state.num_statics

        def enemy_blocked(tile, is_horizontal):
            return (tile == WALL_MID) | (tile == WALL_OBJ)

        def enemy_reflects(tile):
            return (tile == WALL_MID) | (tile == WALL_TOP) | (tile == ENEMY_BARRIER)

        def step_enemy(enemy):
            moved, _ = engine.step_object(
                enemy,
                grid,
                out_of_bounds=jnp.asarray(WALL_MID),
                is_blocked=enemy_blocked,
                will_reflect=enemy_reflects,
                action_vx=jnp.asarray(0),
                action_vy=jnp.asarray(0),
                step_rand_int=step_rand_int,
                max_sub_steps=MAX_SUB_STEPS,
            )
            return moved

        moved_enemies = jax.vmap(step_enemy)(env_state.enemies)
        enemies = jax.tree.map(lambda new, old: jnp.where(enemy_slots, new, old), moved_enemies, env_state.enemies)

        # ---- the agent lands on crates, and only from above ------------------ #
        statics = env_state.statics
        is_crate = static_slots & (statics.type == CRATE)

        def blocked_by_crates(ent, is_horizontal):
            overlap = engine.collides(ent, statics, engine.POS_EPS) & is_crate
            landing = (
                (~is_horizontal)
                & (agent.vy < 0)
                & (action_vy >= 0)
                & (env_state.last_agent_y >= (statics.y + statics.ry + ent.ry))
            )
            hit = (overlap & landing).any()
            return hit, hit

        agent, is_on_crate = engine.step_object(
            agent,
            grid,
            out_of_bounds=jnp.asarray(WALL_MID),
            is_blocked=lambda tile, is_horizontal: (tile == WALL_MID) | (tile == WALL_TOP) | (tile == WALL_OBJ),
            will_reflect=lambda tile: jnp.asarray(False),
            blocked_by_entities=blocked_by_crates,
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # Crates are rigid here instead of pushable (the original's ``push_obj``
        # moves the crate down by the penetration depth), so snap the agent flush
        # on top of the highest crate it landed on.
        crate_top = jnp.max(
            jnp.where(
                is_crate & (jnp.abs(agent.x - statics.x) <= statics.rx + agent.rx),
                statics.y + statics.ry,
                -jnp.inf,
            )
        )
        agent = agent._replace(y=jnp.where(is_on_crate, crate_top + AGENT_RY, agent.y))

        # ---- handle_agent_collision: touching a hazard ends the episode ------ #
        hit_enemy = (engine.collides(agent, enemies, engine.POS_EPS) & enemy_slots & (enemies.type == ENEMY)).any()
        hit_saw = (engine.collides(agent, statics, engine.POS_EPS) & static_slots & (statics.type == SAW)).any()

        # ---- handle_grid_collision: the coin and the lava -------------------- #
        min_x, max_x, min_y, max_y = engine.touched_tiles(agent)
        offsets = jnp.arange(3)
        tile_x = min_x + offsets
        tile_y = min_y + offsets
        tiles = jax.vmap(lambda y: jax.vmap(lambda x: engine.tile_at(grid, x, y, WALL_MID))(tile_x))(tile_y)
        covered = (tile_y[:, None] <= max_y) & (tile_x[None, :] <= max_x)
        touched = jnp.where(covered, tiles, SPACE)

        got_coin = (touched == GOAL).any()
        in_lava = ((touched == LAVA_MID) | (touched == LAVA_TOP)).any()
        reward = jnp.where(got_coin, GOAL_REWARD, 0.0).astype(jnp.float32)
        terminated = got_coin | in_lava | hit_enemy | hit_saw
        terminated = terminated | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)

        stepped = CoinRunState(
            grid=grid,
            agent=agent,
            enemies=enemies,
            statics=statics,
            num_enemies=env_state.num_enemies,
            num_statics=env_state.num_statics,
            steps=env_state.steps + 1,
            key=key,
            last_agent_y=agent.y,
            has_support=has_support,
            is_on_crate=is_on_crate,
            facing_right=facing_right,
        )
        return stepped, reward, terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: CoinRunState,
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

    def observe(self, env_state: CoinRunState) -> jax.Array:
        """Render the agent-centred window with entity layers above the tiles."""
        slots = jnp.arange(MAX_ENTITIES)
        moving = jax.tree.map(
            lambda value: jnp.where(slots < env_state.num_enemies, value, jnp.zeros_like(value)),
            env_state.enemies,
        )
        resting = jax.tree.map(
            lambda value: jnp.where(slots < env_state.num_statics, value, jnp.zeros_like(value)),
            env_state.statics,
        )
        everything = jax.tree.map(
            lambda agent, a, b: jnp.concatenate([agent[None], a, b]),
            env_state.agent,
            moving,
            resting,
        )
        center = jnp.stack([env_state.agent.x, env_state.agent.y])
        return render_scene(env_state.grid, center, everything, VISIBILITY, WALL_MID)
