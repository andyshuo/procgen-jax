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

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

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

#: ``coinrun.cpp``: the sprite sets chosen from per level / per entity.
GROUND_THEMES = ['Dirt', 'Grass', 'Planet', 'Sand', 'Snow', 'Stone']
PLAYER_THEME_COLORS = ['Beige', 'Blue', 'Green', 'Pink', 'Yellow']
WALKING_ENEMIES = [
    'slimeBlock',
    'slimePurple',
    'slimeBlue',
    'slimeGreen',
    'mouse',
    'snail',
    'ladybug',
    'wormGreen',
    'wormPink',
]
CRATE_SPRITES = ['boxCrate', 'boxCrate_double', 'boxCrate_single', 'boxCrate_warning']
#: ``get_adjusted_image_rect`` draws the player taller than its body.
PLAYER_SPRITE_HEIGHT = 1.7415

# Bounds that turn the original's data-dependent loops into static ones.
MAX_SECTIONS = 5  # num_sections = randn(dif) + dif, dif <= 3
MAX_DX = 9  # dx = randn(2*dif) + 3 + dif//3
MAX_PIT = 7  # pit_width <= max_dx
MAX_ENTITIES = 36  # worst case: every section a pit full of enemies
MAX_CRATE_PILE = 3
#: ``int(4*|v|)`` is at most 6 because ``|vx| <= maxspeed`` and ``|vy| <= max_jump``.
MAX_SUB_STEPS = 6


def _tile_sprite_table() -> jax.Array:
    """``(theme, tile id) -> atlas row``, following ``theme_for_grid_obj``."""
    table = jnp.full((len(GROUND_THEMES), 64), sprites.row('_transparent'), jnp.int32)
    for theme, ground in enumerate(GROUND_THEMES):
        table = table.at[theme, WALL_TOP].set(sprites.row(f'Ground/{ground}/mid'))
        table = table.at[theme, WALL_MID].set(sprites.row(f'Ground/{ground}/center'))
    table = table.at[:, GOAL].set(sprites.row('Items/coin'))
    table = table.at[:, LAVA_MID].set(sprites.row('Tiles/lava'))
    table = table.at[:, LAVA_TOP].set(sprites.row('Tiles/lava_top'))
    # ENEMY_BARRIER is invisible, as in the original (``image_for_type`` returns -1).
    return table


def _entity_sprite_table() -> jax.Array:
    """``(image_type, image_theme) -> atlas row``, following ``asset_for_type``."""
    table = jnp.full((32, 10), sprites.row('_transparent'), jnp.int32)
    for image_type, pose in (
        (engine.PLAYER, 'stand'),
        (PLAYER_JUMP, 'jump'),
        (PLAYER_RIGHT1, 'walk1'),
        (PLAYER_RIGHT2, 'walk2'),
    ):
        for theme, colour in enumerate(PLAYER_THEME_COLORS):
            table = table.at[image_type, theme].set(sprites.row(f'Players/{colour}/{pose}'))
    for theme, enemy in enumerate(WALKING_ENEMIES):
        table = table.at[ENEMY1, theme].set(sprites.row(f'Enemies/{enemy}/idle'))
        table = table.at[ENEMY2, theme].set(sprites.row(f'Enemies/{enemy}/move'))
    table = table.at[SAW, 0].set(sprites.row('Tiles/saw/idle'))
    table = table.at[SAW2, 0].set(sprites.row('Tiles/saw/move'))
    for theme, crate in enumerate(CRATE_SPRITES):
        table = table.at[CRATE, theme].set(sprites.row(f'Tiles/{crate}'))
    return table


TILE_SPRITES = _tile_sprite_table()
ENTITY_SPRITES = _entity_sprite_table()


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
    wall_theme: jax.Array


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


def _push_enemy(pool, count, x, y, vx, theme, active):
    ent = engine.entity(
        x + 0.5,
        y + 0.5,
        vx,
        0.0,
        type=ENEMY,
        image_type=ENEMY1,
        image_theme=theme,
        render_z=1,
        smart_step=True,
    )
    return _append(pool, count, ent, active)


def _push_crate(pool, count, x, y, theme, active):
    ent = engine.entity(x + 0.5, y + 0.5, type=CRATE, image_theme=theme)
    return _append(pool, count, ent, active)


def generate_level(level_seed: jax.Array):
    """``CoinRun::game_reset`` + ``generate_coin_to_the_right``, bit for bit.

    Every random draw happens in the same order as the C++, including the draws
    that the original's short-circuits skip: those are gated with ``lax.cond`` so a
    skipped draw does not advance the generator. That is what makes a level seed
    mean the same level as the reference implementation (see
    ``tools/coinrun_reference/coinrun_level.cpp``).

    Returns ``(grid, enemies, statics, num_enemies, num_statics, wall_theme,
    player_colour)``.
    """
    state = rng.seed(level_seed.astype(jnp.uint32))

    # ---- BasicAbstractGame::game_reset and CoinRun::game_reset ----------------- #
    state, _ = rng.rand01(state)  # bg_pct_x
    state, _ = rng.randn(state, 8)  # background_index (platform_backgrounds.size())
    state, _ = rng.rand01(state)  # random agent start, x
    state, _ = rng.rand01(state)  # random agent start, y
    state, player_colour = rng.randn(state, len(PLAYER_THEME_COLORS))
    state, wall_theme = rng.randn(state, len(GROUND_THEMES))

    # ---- generate_coin_to_the_right ------------------------------------------- #
    state, dif_draw = rng.randn(state, 3)
    difficulty = dif_draw + 1
    state, section_draw = rng.randn(state, difficulty)
    num_sections = section_draw + difficulty
    state, danger_type = rng.randn(state, 3)

    max_dy = jnp.asarray(int(MAX_JUMP * MAX_JUMP / (2 * GRAVITY) - 0.5), jnp.int32)
    max_dx = jnp.asarray(int(MAX_SPEED * 2 * MAX_JUMP / GRAVITY - 0.5), jnp.int32)
    int_dif = difficulty // 3

    grid = jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32)
    grid = fill(grid, 0, 0, WORLD_DIM, 1, WALL_TOP)
    grid = fill(grid, 0, 0, 1, WORLD_DIM, WALL_MID)
    grid = fill(grid, WORLD_DIM - 1, 0, 1, WORLD_DIM, WALL_MID)
    grid = fill(grid, 0, WORLD_DIM - 1, WORLD_DIM, 1, WALL_MID)

    enemies = empty_entity_pool(MAX_ENTITIES)
    statics = empty_entity_pool(MAX_ENTITIES)

    def draw(state, active, fn):
        """Consume a draw only when ``active`` -- the original's short-circuits."""
        return jax.lax.cond(active, lambda s: fn(s), lambda s: (s, jnp.zeros((), jnp.int32)), state)

    def body(carry, section):
        grid, cx, cy, enemies, n_enemies, statics, n_statics, state = carry
        active = (section < num_sections) & (cx + 15 < WORLD_DIM)

        # ---- where this section goes ------------------------------------------ #
        state, dy_draw = draw(state, active, lambda s: rng.randn(s, 4))
        dy = dy_draw + 1 + int_dif
        dy = jnp.minimum(dy, max_dy)
        high = cy >= 20
        state, flip_draw = draw(state, active & (cy >= 5) & ~high, lambda s: rng.randn(s, 2))
        dy = jnp.where(high | ((cy >= 5) & (flip_draw == 1)), -dy, dy)
        state, dx_draw = draw(state, active, lambda s: rng.randn(s, 2 * difficulty))
        dx = dx_draw + 3 + int_dif
        cy = jnp.where(active, jnp.maximum(cy + dy, 1), cy)

        pit_gate = active & (dx > 7) & (cy > 3)
        state, pit_draw = draw(state, pit_gate, lambda s: rng.randn(s, 20))
        use_pit = pit_gate & (pit_draw >= difficulty)

        # ---- pit branch -------------------------------------------------------- #
        state, x1_draw = draw(state, use_pit, lambda s: rng.randn(s, 3))
        state, x2_draw = draw(state, use_pit, lambda s: rng.randn(s, 3))
        x1 = x1_draw + 1
        x2 = x2_draw + 1
        too_wide = (dx - x1 - x2) > max_dx
        pit_width = jnp.where(too_wide, max_dx, dx - x1 - x2)
        x2 = jnp.where(too_wide, dx - x1 - pit_width, x2)

        pit = _fill_ground(grid, cx, 0, x1, cy)
        pit = _fill_ground(pit, cx + dx - x2, 0, x2, cy)
        state, lava_draw = draw(state, use_pit, lambda s: rng.randn(s, cy - 3))
        lava_height = lava_draw + 1
        pit = jnp.where(
            use_pit & (danger_type == 0),
            _fill_lava(pit, cx + x1, 1, pit_width, lava_height),
            pit,
        )

        # Hazards along the pit floor: saws cost no draw, enemies cost two each.
        for step_index in range(MAX_PIT):
            in_pit = use_pit & (step_index < pit_width)
            statics, n_statics = _push_saw(statics, n_statics, cx + x1 + step_index, 1, in_pit & (danger_type == 1))
            spawning = in_pit & (danger_type == 2)
            state, vx_draw = draw(state, spawning, lambda s: rng.randn(s, 2))
            state, theme = draw(state, spawning, lambda s: rng.randn(s, len(WALKING_ENEMIES)))
            enemies, n_enemies = _push_enemy(
                enemies,
                n_enemies,
                cx + x1 + step_index,
                1,
                0.15 * (2 * vx_draw - 1),
                theme,
                spawning,
            )

        # A bridge over wide pits: 5 and 6 tile pits take a shelf, wider ones a span.
        shallow = use_pit & (pit_width == 5)
        middle = use_pit & (pit_width == 6)
        deep = use_pit & (pit_width >= 7)
        state, a = draw(state, shallow, lambda s: rng.randn(s, 2))
        state, b = draw(state, shallow, lambda s: rng.randn(s, 2))
        state, c = draw(state, middle, lambda s: rng.randn(s, 2))
        state, d = draw(state, middle, lambda s: rng.randn(s, 2))
        state, e = draw(state, deep, lambda s: rng.randn(s, 2))
        state, f = draw(state, deep, lambda s: rng.randn(s, 2))
        bridge_x = jnp.where(shallow, 1 + a, jnp.where(middle, 2 + c, 2 + e))
        bridge_w = jnp.where(shallow, 1 + b, jnp.where(middle, 1 + d, pit_width - bridge_x - (2 + f)))
        pit = jnp.where(
            use_pit & (pit_width > 4),
            _fill_ground(pit, cx + x1 + bridge_x, cy - 1, bridge_w, 1),
            pit,
        )

        # ---- flat run ---------------------------------------------------------- #
        flat = _fill_ground(grid, cx, 0, dx, cy)
        on_flat = active & ~use_pit
        state, saw_gate = draw(state, on_flat, lambda s: rng.randn(s, 10))
        saw_on = (saw_gate < 2 * difficulty) & (dx > 3)
        state, ob1_draw = draw(state, on_flat & saw_on, lambda s: rng.randn(s, dx - 2))
        ob1_x = jnp.where(saw_on, cx + ob1_draw + 1, -1)
        statics, n_statics = _push_saw(statics, n_statics, ob1_x, cy, on_flat & saw_on)

        state, enemy_gate = draw(state, on_flat, lambda s: rng.randn(s, 10))
        enemy_on = (enemy_gate < difficulty) & (dx > 3) & (max_dx >= 4)
        state, ob2_draw = draw(state, on_flat & enemy_on, lambda s: rng.randn(s, dx - 2))
        ob2_x = jnp.where(enemy_on, cx + ob2_draw + 1, -1)
        state, enemy_vx_draw = draw(state, on_flat & enemy_on, lambda s: rng.randn(s, 2))
        state, enemy_theme = draw(state, on_flat & enemy_on, lambda s: rng.randn(s, len(WALKING_ENEMIES)))
        enemies, n_enemies = _push_enemy(
            enemies,
            n_enemies,
            ob2_x,
            cy,
            0.15 * (2 * enemy_vx_draw - 1),
            enemy_theme,
            on_flat & enemy_on,
        )

        for _ in range(2):
            state, crate_draw = draw(state, on_flat, lambda s: rng.randn(s, dx - 2))
            crate_x = cx + crate_draw + 1
            state, coin = draw(state, on_flat, lambda s: rng.randn(s, 2))
            crate_on = on_flat & (coin == 1) & (crate_x != ob1_x) & (crate_x != ob2_x)
            state, pile_draw = draw(state, crate_on, lambda s: rng.randn(s, 3))
            pile_height = pile_draw + 1
            for level_index in range(MAX_CRATE_PILE):
                stacking = crate_on & (level_index < pile_height)
                state, crate_theme = draw(state, stacking, lambda s: rng.randn(s, len(CRATE_SPRITES)))
                statics, n_statics = _push_crate(statics, n_statics, crate_x, cy + level_index, crate_theme, stacking)

        # ---- carve this section, then fence it -------------------------------- #
        grid = jnp.where(active, jnp.where(use_pit, pit, flat), grid)
        previous = engine.tile_at(grid, cx - 1, cy, WALL_MID)
        grid = jnp.where(
            active & ~((previous == WALL_MID) | (previous == WALL_TOP)),
            fill(grid, cx - 1, cy, 1, 1, ENEMY_BARRIER),
            grid,
        )
        cx = jnp.where(active, cx + dx, cx)
        grid = jnp.where(active, fill(grid, cx, cy, 1, 1, ENEMY_BARRIER), grid)

        return (grid, cx, cy, enemies, n_enemies, statics, n_statics, state), None

    carry = (
        grid,
        jnp.asarray(5, jnp.int32),
        jnp.asarray(1, jnp.int32),
        enemies,
        jnp.asarray(0, jnp.int32),
        statics,
        jnp.asarray(0, jnp.int32),
        state,
    )
    carry, _ = jax.lax.scan(body, carry, jnp.arange(MAX_SECTIONS))
    grid, cx, cy, enemies, n_enemies, statics, n_statics, _ = carry

    # The coin sits on a plinth one tile below it; the rest of the world is wall.
    grid = fill(grid, cx, cy, 1, 1, GOAL)
    grid = _fill_ground(grid, cx, 0, 1, cy)
    grid = fill(grid, cx + 1, 0, WORLD_DIM - cx - 1, WORLD_DIM, WALL_MID)
    return grid, enemies, statics, n_enemies, n_statics, wall_theme, player_colour


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class CoinRunEnv(Env):
    """ProcGen's ``coinrun`` as a functional, jit-friendly environment."""

    def __init__(self, timeout: int = TIMEOUT, level_seed: int | None = None):
        self.timeout = timeout
        #: With a seed the env replays exactly that level, forever; it is the same
        #: level the reference implementation builds for the same seed.  Without
        #: one, each episode draws its own seed from the episode key.
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
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        grid, enemies, statics, num_enemies, num_statics, wall_theme, player_colour = generate_level(level_seed)
        return CoinRunState(
            grid=grid,
            agent=engine.entity(
                START_X,
                START_Y,
                rx=AGENT_RX,
                ry=AGENT_RY,
                type=engine.PLAYER,
                image_theme=player_colour,
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
            wall_theme=wall_theme,
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

        # ---- image_for_type: pick the animation frame the renderer shows ---- #
        steps = env_state.steps
        enemy_frame = jnp.where((steps // 5) % 2 == 0, ENEMY1, ENEMY2)
        enemies = enemies._replace(
            image_type=jnp.broadcast_to(enemy_frame, env_state.enemies.image_type.shape),
            is_reflected=enemies.vx > 0,
        )
        saw_frame = jnp.where(steps % 2 == 0, SAW, SAW2)
        statics = statics._replace(image_type=jnp.where(statics.type == SAW, saw_frame, statics.image_type))
        idle = (jnp.abs(agent.vx) < 0.01) & (action_vx == 0) & has_support
        walking = ((steps // 5) % 2 == 0) | ~has_support
        agent = agent._replace(
            image_type=jnp.where(idle, engine.PLAYER, jnp.where(walking, PLAYER_RIGHT1, PLAYER_RIGHT2)),
            is_reflected=~facing_right,
        )

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
            wall_theme=env_state.wall_theme,
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
        """Render the agent-centred window with sprite layers above the tiles."""
        bodies = scene_entities(env_state)
        is_player = jnp.isin(
            bodies.image_type,
            jnp.asarray([engine.PLAYER, PLAYER_JUMP, PLAYER_RIGHT1, PLAYER_RIGHT2]),
        )
        center = jnp.stack([env_state.agent.x, env_state.agent.y])
        return render_textured(
            env_state.grid,
            center,
            bodies,
            TILE_SPRITES[env_state.wall_theme],
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.where(is_player, PLAYER_SPRITE_HEIGHT, 1.0),
            sprites.PIXELS,
            VISIBILITY,
            WALL_MID,
        )


def scene_entities(env_state: CoinRunState) -> engine.Entity:
    """Every body of the level as one batched ``Entity``, for the renderer.

    Inactive pool slots are zeroed so they cannot paint anything; the agent comes
    first so its layer wins ties.
    """
    slots = jnp.arange(MAX_ENTITIES)
    moving = jax.tree.map(
        lambda value: jnp.where(slots < env_state.num_enemies, value, jnp.zeros_like(value)),
        env_state.enemies,
    )
    resting = jax.tree.map(
        lambda value: jnp.where(slots < env_state.num_statics, value, jnp.zeros_like(value)),
        env_state.statics,
    )
    return jax.tree.map(
        lambda agent, a, b: jnp.concatenate([agent[None], a, b]),
        env_state.agent,
        moving,
        resting,
    )
