"""A pure-JAX port of ProcGen's ``jumper``, from ``procgen/src/games/jumper.cpp``.

A cave platformer: the level is a no-dead-ends maze scaled up three times, roughened
by two cellular-automata passes, reduced to its largest room, and then carved back
open along the path from the cave floor to the carrot.  The rabbit can jump twice
before it has to land, and the spikes cost you the level.

The generator is the reason this port exists: it is the only ProcGen game that uses
the whole of :mod:`procgen_jax.mazegen` and :mod:`procgen_jax.roomgen`.

Deviations: the trail the rabbit leaves and the compass HUD are not drawn (both are
pure decoration), and the sprite does not rotate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, mazegen, rng, roomgen, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``jumper.cpp`` object ids.
GOAL = 1
SPIKE = 2
CAVEWALL = 6
CAVEWALL_TOP = 7
PLAYER_JUMP = 9
PLAYER_LEFT1 = 10
PLAYER_LEFT2 = 11
PLAYER_RIGHT1 = 12
PLAYER_RIGHT2 = 13
#: ProcGen's own ids, which the maze and room generators speak.
MAZE_WALL = 51
SPACE = 100

GOAL_REWARD = 10.0
WORLD_DIM = 40
MAZE_SCALE = 3
MAZE_DIM = WORLD_DIM // MAZE_SCALE
VIEW_TILES = 16
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.5
GRAVITY = 0.15
#: ``if (agent->vy > -2) agent->vy -= 0.15f``
TERMINAL_VELOCITY = -2.0
JUMP_COOLDOWN = 3
JUMPS_PER_LANDING = 2
NUM_WALL_THEMES = 4
AGENT_RX = 0.254
AGENT_RY = 0.4
SPIKE_RX = 0.23
SPIKE_RY = 0.4
GOAL_RADIUS = 0.5
SPIKE_PROB = 0.2
WIDEN_ROUNDS = 4
#: The carrot, a couple of hundred spikes at most, and room to spare.
MAX_ENTITIES = 256
MAX_SUB_STEPS = 6

WALL_TOP_SPRITES = ['tileBlue_05', 'tileGreen_05', 'tileYellow_06', 'tileBrown_06']
WALL_SPRITES = ['tileBlue_08', 'tileGreen_08', 'tileYellow_09', 'tileBrown_09']

ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[GOAL, 0]
    .set(sprites.row('misc/jumper/carrot'))
    .at[SPIKE, 0]
    .set(sprites.row('misc/jumper/spikeMan_stand'))
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/jumper/bunny2_ready'))
    .at[PLAYER_JUMP, 0]
    .set(sprites.row('misc/jumper/bunny2_jump'))
    .at[PLAYER_LEFT1, 0]
    .set(sprites.row('misc/jumper/bunny2_walk1'))
    .at[PLAYER_LEFT2, 0]
    .set(sprites.row('misc/jumper/bunny2_walk2'))
    .at[PLAYER_RIGHT1, 0]
    .set(sprites.row('misc/jumper/bunny2_walk1'))
    .at[PLAYER_RIGHT2, 0]
    .set(sprites.row('misc/jumper/bunny2_walk2'))
)
#: ``TILE_ROWS`` is indexed by tile id, and the generator uses ``SPACE`` = 100, so the
#: table has to reach past it; the wall sprites are filled in per level's theme.
TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32)
    .at[CAVEWALL]
    .set(sprites.row('platformer/tileBlue_08'))
    .at[CAVEWALL_TOP]
    .set(sprites.row('platformer/tileBlue_05'))
)
WALL_ROWS = jnp.asarray([sprites.row(f'platformer/{name}') for name in WALL_SPRITES], jnp.int32)
WALL_TOP_ROWS = jnp.asarray([sprites.row(f'platformer/{name}') for name in WALL_TOP_SPRITES], jnp.int32)
#: The original paints a cave background; this is a flat stand-in.
CAVE_COLOUR = (30, 26, 34)


class JumperState(NamedTuple):
    """Game state: the cave, the rabbit, the carrot and the spikes."""

    grid: jax.Array  # (WORLD_DIM * WORLD_DIM,), ProcGen ids: CAVEWALL, CAVEWALL_TOP, SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) the carrot and the spikes
    wall_theme: jax.Array
    has_support: jax.Array
    jump_count: jax.Array
    jump_time: jax.Array
    facing_right: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def _padded_view(flat: jax.Array):
    """The grid padded with ``MAZE_WALL`` (the out-of-bounds tile) as ``[y][x]``."""
    return jnp.pad(flat.reshape(WORLD_DIM, WORLD_DIM), 1, constant_values=MAZE_WALL)


def _space_on_ground(flat: jax.Array) -> jax.Array:
    """``is_space_on_ground`` for every cell, as a flat mask.

    A cell counts when it and the tile above are open and the tile below is a cave
    wall (or off the grid, which the padding turns into a wall).
    """
    view = _padded_view(flat)
    rows = jnp.arange(WORLD_DIM)
    here = view[1:-1, 1:-1]
    above = view[rows + 2, 1:-1]
    below = view[rows, 1:-1]
    return ((here == SPACE) & (above == SPACE) & ((below == CAVEWALL) | (below == MAZE_WALL))).reshape(-1)


class JumperEnv(Env):
    """ProcGen's ``jumper`` as a functional, jit-friendly environment."""

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

        def reset(env_state: JumperState) -> JumperState:
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
    def reset_env_state(self, key: jax.Array) -> JumperState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y
        state, wall_theme = rng.randn(state, NUM_WALL_THEMES)

        state, maze = mazegen.generate_maze_no_dead_ends(state, MAZE_DIM)
        size = WORLD_DIM * WORLD_DIM

        # ---- the scaled maze, roughened up ---------------------------------- #
        # ``obj = maze[(y / 3) + 1][(x / 3) + 1]`` then one rand01 per cell.
        cells = jnp.arange(size, dtype=jnp.int32)
        cell_x = cells % WORLD_DIM
        cell_y = cells // WORLD_DIM
        source = maze[(cell_y // MAZE_SCALE) + 1, (cell_x // MAZE_SCALE) + 1]
        wall_prob = jnp.where(source == MAZE_WALL, 0.8, 0.2)

        def fill(index, carry):
            state, flat = carry
            state, draw = rng.rand01(state)
            flat = flat.at[index].set(jnp.where(draw < wall_prob[index], MAZE_WALL, SPACE))
            return state, flat

        state, flat = jax.lax.fori_loop(0, size, fill, (state, jnp.full((size,), SPACE, jnp.int32)))

        flat = roomgen.update(flat, WORLD_DIM)
        flat = roomgen.update(flat, WORLD_DIM)

        # A solid border, so the room search and the path stay inside.
        border = jnp.zeros((WORLD_DIM, WORLD_DIM), bool)
        border = border.at[0, :].set(True).at[-1, :].set(True).at[:, 0].set(True).at[:, -1].set(True)
        flat = jnp.where(border.reshape(-1), CAVEWALL, flat)

        best_room = roomgen.find_best_room(flat, WORLD_DIM)
        count = best_room.sum()
        free_cells = jnp.sort(jnp.where(best_room, cells, size))
        state, goal_draw = rng.randn(state, count)
        goal_cell = free_cells[goal_draw]

        # Everything but the best room becomes wall, then the agent picks a footing.
        flat = jnp.where(best_room, SPACE, CAVEWALL)
        candidates = _space_on_ground(flat)
        candidate_count = candidates.sum()
        candidate_cells = jnp.sort(jnp.where(candidates, cells, size))
        state, agent_draw = rng.randn(state, candidate_count)
        agent_cell = candidate_cells[agent_draw]

        # ---- the path, widened, is the only floor that survives ------------- #
        path_cells, path_length = roomgen.find_path(flat, WORLD_DIM, agent_cell, goal_cell)
        # the path comes back padded, and a traced length cannot slice, so mask it
        on_path = jnp.arange(path_cells.shape[0]) < path_length
        wide = jnp.zeros((size + 1,), bool).at[jnp.where(on_path, path_cells, size)].set(True)[:size]
        wide = roomgen.expand_room(flat, WORLD_DIM, wide, WIDEN_ROUNDS)
        flat = jnp.where(wide, SPACE, CAVEWALL)

        # ---- spikes ---------------------------------------------------------- #
        # One gated draw per cell that has open ground on both sides, and the cell
        # scanned becomes a spike straight away, so later checks see it.
        def spike_scan(index, carry):
            state, flat = carry
            ground = _space_on_ground(flat)
            left = index % WORLD_DIM
            right_edge = left - 1
            right = left + 1
            has_left = (right_edge >= 0) & ground[jnp.clip(index - 1, 0, size - 1)]
            has_right = (right <= WORLD_DIM - 1) & ground[jnp.clip(index + 1, 0, size - 1)]
            qualifies = ground[index] & has_left & has_right
            state, draw = jax.lax.cond(
                qualifies,
                lambda s: rng.rand01(s),
                lambda s: (s, jnp.ones((), jnp.float32)),
                state,
            )
            spike = qualifies & (draw < SPIKE_PROB)
            flat = flat.at[index].set(jnp.where(spike, SPIKE, flat[index]))
            return state, flat

        state, flat = jax.lax.fori_loop(0, size, spike_scan, (state, flat))

        # ---- keep the cave from growing long vertical walls ------------------ #
        def prune(index, carry):
            state, flat = carry
            view = _padded_view(flat)
            row = index // WORLD_DIM
            column = index % WORLD_DIM

            def left_wall(at_row):
                cell = view[at_row, column + 1]
                right = view[at_row, column + 2]
                return (cell == CAVEWALL) & (right == SPACE)

            def right_wall(at_row):
                cell = view[at_row, column + 1]
                left = view[at_row, column]
                return (cell == CAVEWALL) & (left == SPACE)

            left_column = left_wall(row + 1) & left_wall(row + 2) & left_wall(row + 3)
            state, left_draw = jax.lax.cond(
                left_column,
                lambda st: rng.randn(st, 3),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            target_row = row + left_draw
            target = jnp.clip(target_row * WORLD_DIM + column, 0, size - 1)
            flat = flat.at[target].set(jnp.where(left_column & (target_row < WORLD_DIM), SPACE, flat[target]))

            right_column = right_wall(row + 1) & right_wall(row + 2) & right_wall(row + 3)
            state, right_draw = jax.lax.cond(
                right_column,
                lambda st: rng.randn(st, 3),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            target_row = row + right_draw
            target = jnp.clip(target_row * WORLD_DIM + column, 0, size - 1)
            flat = flat.at[target].set(jnp.where(right_column & (target_row < WORLD_DIM), SPACE, flat[target]))
            return state, flat

        state, flat = jax.lax.fori_loop(0, size, prune, (state, flat))

        # ---- the carrot, the spikes and the rabbit --------------------------- #
        agent_x = (agent_cell % WORLD_DIM).astype(jnp.float32) + 0.5
        agent_y = (agent_cell // WORLD_DIM).astype(jnp.float32) + AGENT_RY

        spike_cells = flat == SPIKE
        # the spike tiles become floor again; the spikes become bodies
        flat = jnp.where(spike_cells, SPACE, flat)

        pool = empty_pool()
        goal = engine.entity(
            (goal_cell % WORLD_DIM).astype(jnp.float32) + 0.5,
            (goal_cell // WORLD_DIM).astype(jnp.float32) + 0.5,
            rx=GOAL_RADIUS,
            ry=GOAL_RADIUS,
            type=GOAL,
            image_type=GOAL,
            active=True,
        )
        pool = jax.tree.map(lambda buf, value: buf.at[0].set(value), pool, goal)

        spike_indices = jnp.sort(jnp.where(spike_cells, cells, size))
        spike_count = spike_cells.sum()

        def place_spike(slot, carry):
            pool = carry
            cell = spike_indices[jnp.clip(slot, 0, size - 1)]
            spike = engine.entity(
                (cell % WORLD_DIM).astype(jnp.float32) + 0.5,
                (cell // WORLD_DIM).astype(jnp.float32) + SPIKE_RY,
                rx=SPIKE_RX,
                ry=SPIKE_RY,
                type=SPIKE,
                image_type=SPIKE,
                active=True,
            )
            return jax.tree.map(lambda buf, value, s=slot + 1: buf.at[s].set(value), pool, spike)

        pool = jax.lax.fori_loop(0, MAX_ENTITIES - 1, place_spike, pool)
        filled = jnp.arange(MAX_ENTITIES) <= spike_count
        blank = empty_pool()
        pool = jax.tree.map(lambda buf, zero: jnp.where(filled, buf, zero), pool, blank)

        # ---- cave wall tops --------------------------------------------------- #
        view = _padded_view(flat)
        top_walls = (view[1:-1, 1:-1] == CAVEWALL) & (view[2:, 1:-1] == SPACE)
        flat = jnp.where(top_walls.reshape(-1), CAVEWALL_TOP, flat)

        agent = engine.entity(
            agent_x,
            agent_y,
            rx=AGENT_RX,
            ry=AGENT_RY,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            is_reflected=False,
            render_z=1,
            smart_step=True,
        )
        return JumperState(
            grid=flat,
            agent=agent,
            entities=pool,
            wall_theme=wall_theme,
            has_support=jnp.asarray(True),
            jump_count=jnp.asarray(0, jnp.int32),
            jump_time=jnp.asarray(0, jnp.int32),
            facing_right=jnp.asarray(True),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: JumperState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1
        agent = env_state.agent
        grid = env_state.grid

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = jnp.maximum(move_action % 3 - 1, 0)

        # ---- set_action_xy: footing, then the jump budget --------------------- #
        view = _padded_view(grid)
        below = []
        for offset in (AGENT_RX - 0.01, -(AGENT_RX - 0.01)):
            cell_x = jnp.floor(agent.x + offset).astype(jnp.int32)
            cell_y = jnp.floor(agent.y - (AGENT_RY + 0.01)).astype(jnp.int32)
            tile = view[jnp.clip(cell_y, 0, WORLD_DIM - 1) + 1, jnp.clip(cell_x, 0, WORLD_DIM - 1) + 1]
            below.append(tile)
        support = jnp.stack(below)
        can_support = (support == CAVEWALL) | (support == CAVEWALL_TOP) | (support == MAZE_WALL)
        has_support = can_support.any()
        jump_count = jnp.where(has_support, JUMPS_PER_LANDING, env_state.jump_count)
        can_jump = (action_vy == 1) & (jump_count > 0) & ((now - env_state.jump_time) > JUMP_COOLDOWN)
        jump_count = jnp.where(can_jump, jump_count - 1, jump_count)
        action_vy = jnp.where(can_jump, action_vy, 0)
        jump_time = jnp.where(action_vy > 0, now, env_state.jump_time)
        facing_right = jnp.where(action_vx > 0, True, jnp.where(action_vx < 0, False, env_state.facing_right))

        # ---- the base velocity model ----------------------------------------- #
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy, is_reflected=action_vx < 0)

        agent, _ = engine.step_object(
            agent,
            grid.reshape(WORLD_DIM, WORLD_DIM),
            out_of_bounds=jnp.asarray(CAVEWALL),
            is_blocked=lambda tile, is_horizontal: (tile == CAVEWALL) | (tile == CAVEWALL_TOP),
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        # ---- contact: the carrot wins, a spike ends the run ------------------- #
        touching = engine.collides(agent, pool) & pool.active
        reached_goal = (touching & (pool.type == GOAL)).any()
        hit_spike = (touching & (pool.type == SPIKE)).any()
        reward = jnp.where(reached_goal, GOAL_REWARD, 0.0)
        terminated = reached_goal | hit_spike

        # ---- gravity, applied after the move ---------------------------------- #
        agent = agent._replace(vy=jnp.where(agent.vy > TERMINAL_VELOCITY, agent.vy - GRAVITY, agent.vy))

        # ---- the sprite the original would pick -------------------------------- #
        idle = (jnp.abs(agent.vx) < 0.01) & (action_vx == 0) & has_support
        walk_frame = now // 5 % 2 == 0
        right_sprite = jnp.where(walk_frame | ~has_support, PLAYER_RIGHT1, PLAYER_RIGHT2)
        left_sprite = jnp.where(walk_frame | ~has_support, PLAYER_LEFT1, PLAYER_LEFT2)
        image_type = jnp.where(idle, engine.PLAYER, jnp.where(facing_right, right_sprite, left_sprite))

        stepped = JumperState(
            grid=grid,
            agent=agent._replace(image_type=image_type),
            entities=pool,
            wall_theme=env_state.wall_theme,
            has_support=has_support,
            jump_count=jump_count,
            jump_time=jump_time,
            facing_right=facing_right,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: JumperState,
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

    def observe(self, env_state: JumperState) -> jax.Array:
        """``visibility = 16`` around the rabbit, with the level's cave theme."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        theme = env_state.wall_theme
        tile_rows = TILE_ROWS.at[CAVEWALL].set(WALL_ROWS[theme]).at[CAVEWALL_TOP].set(WALL_TOP_ROWS[theme])
        return render_textured(
            env_state.grid.reshape(WORLD_DIM, WORLD_DIM),
            jnp.stack([env_state.agent.x, env_state.agent.y]),
            bodies,
            tile_rows,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            VIEW_TILES,
            jnp.asarray(CAVEWALL),
            background=CAVE_COLOUR,
        )
