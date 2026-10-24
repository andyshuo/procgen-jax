"""A pure-JAX port of ProcGen's ``chaser``, from ``procgen/src/games/chaser.cpp``.

Pac-Man in a ProcGen maze: every open cell holds an orb, four quadrants hide one
crystal each, and the crystals make the ghosts edible for seventy-five steps -- while
you are dangerous they flee, otherwise they hunt you along the maze.

Deviations: orbs are a flat green tile (the original paints a rectangle instead of
using a sprite), and ``step_rand_int`` comes from a JAX key; chaser draws nothing else
during a step, so the level itself is unaffected.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, mazegen, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``chaser.cpp`` object ids.
LARGE_ORB = 2
ENEMY_WEAK = 3
ENEMY_EGG = 4
MAZE_WALL = 5
ENEMY = 6
#: This port's small tile ids for the orb and the temporary placement marker.
ORB_TILE = 9
MARKER_TILE = 10
#: ProcGen's wall id, which the maze generator speaks.
MAZE_WALL_SOURCE = 51

ORB_REWARD = 0.04
COMPLETION_BONUS = 10.0
TIMEOUT = 1000
NUM_ACTIONS = 15
MAX_SPEED = 0.5
#: HardMode's numbers.
MAZE_DIM = 13
TOTAL_ENEMIES = 3
EAT_TIMEOUT = 75
EGG_TIMEOUT = 50
AGENT_RADIUS = 0.5
ORB_RADIUS = 0.4
ENEMY_RADIUS = 0.5
MAX_ENTITIES = 32
MAX_SUB_STEPS = 6
NUM_QUADRANTS = 4

TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32)
    .at[MAZE_WALL]
    .set(sprites.row('misc/chaser/tileStone_slope'))
    .at[ORB_TILE]
    .set(sprites.row('misc/chaser/orb'))
)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/chaser/enemyFloating_1b'))
    .at[LARGE_ORB, 0]
    .set(sprites.row('misc/chaser/yellowCrystal'))
    .at[ENEMY_WEAK, 0]
    .set(sprites.row('misc/chaser/enemyWalking_1b'))
    .at[ENEMY_EGG, 0]
    .set(sprites.row('misc/chaser/enemySpikey_1b'))
    .at[ENEMY, 0]
    .set(sprites.row('misc/chaser/enemyFlying_1'))
    .at[ENEMY + 1, 0]
    .set(sprites.row('misc/chaser/enemyFlying_2'))
    .at[ENEMY + 2, 0]
    .set(sprites.row('misc/chaser/enemyFlying_3'))
)
#: The original paints a top-down floor; this is a flat stand-in.
FLOOR_COLOUR = (26, 26, 34)


class ChaserState(NamedTuple):
    """Game state: the orb-filled maze, the eye, and the ghosts hunting it."""

    grid: jax.Array  # (MAZE_DIM * MAZE_DIM,) tiles: MAZE_WALL, SPACE, ORB_TILE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) crystals, eggs and ghosts
    free_cells: jax.Array  # (MAZE_DIM * MAZE_DIM,) the open cells, ascending, padded
    free_count: jax.Array
    is_space: jax.Array  # (MAZE_DIM * MAZE_DIM,) bool
    eat_time: jax.Array
    total_orbs: jax.Array
    orbs_collected: jax.Array
    total_enemies: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def _simple_choose(state, size, count):
    """``simple_choose``: ``count`` distinct draws, redrawing while one repeats."""
    chosen = jnp.zeros((count,), jnp.int32)
    taken = jnp.zeros((MAZE_DIM * MAZE_DIM,), bool)
    for pick in range(count):

        def body(carry, taken_at=taken):
            state, _, _ = carry
            state, draw = rng.randn(state, jnp.maximum(size, 1))
            return state, draw, taken_at[jnp.clip(draw, 0, MAZE_DIM * MAZE_DIM - 1)]

        state, value, _ = jax.lax.while_loop(
            lambda carry: carry[2], body, (state, jnp.zeros((), jnp.int32), jnp.asarray(True))
        )
        chosen = chosen.at[pick].set(value)
        taken = taken.at[jnp.clip(value, 0, MAZE_DIM * MAZE_DIM - 1)].set(True)
    return state, chosen


class ChaserEnv(Env):
    """ProcGen's ``chaser`` as a functional, jit-friendly environment."""

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

        def reset(env_state: ChaserState) -> ChaserState:
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
    def reset_env_state(self, key: jax.Array) -> ChaserState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        state, maze = mazegen.generate_maze_no_dead_ends(state, MAZE_DIM)
        state, extra_quadrant = rng.randn(state, NUM_QUADRANTS)

        size = MAZE_DIM * MAZE_DIM
        cells = jnp.arange(size, dtype=jnp.int32)
        cell_x = cells % MAZE_DIM
        cell_y = cells // MAZE_DIM
        maze_flat = maze.reshape(-1)
        source = maze_flat[(cell_y + 1) * (MAZE_DIM + 2) + (cell_x + 1)]
        grid = jnp.where(source == MAZE_WALL_SOURCE, MAZE_WALL, SPACE)

        # ---- one crystal per quadrant, except the drawn one ------------------ #
        quadrant = ((cell_x >= MAZE_DIM / 2).astype(jnp.int32) * 2) + (cell_y >= MAZE_DIM / 2).astype(jnp.int32)
        crystal_cells = jnp.full((NUM_QUADRANTS,), size, jnp.int32)
        for index in range(NUM_QUADRANTS):
            inside = (grid == SPACE) & (quadrant == index)
            inside_count = inside.sum()
            wanted = index != extra_quadrant
            state, picked = jax.lax.cond(
                wanted & (inside_count > 0),
                lambda st, count=inside_count: _simple_choose(st, count, 1),
                lambda st: (st, jnp.zeros((1,), jnp.int32)),
                state,
            )
            cell_list = jnp.sort(jnp.where(inside, cells, size))
            chosen = cell_list[jnp.clip(picked[0], 0, size - 1)]
            crystal_cells = crystal_cells.at[index].set(jnp.where(wanted & (inside_count > 0), chosen, size))
            grid = grid.at[jnp.clip(chosen, 0, size - 1)].set(
                jnp.where(wanted & (inside_count > 0), MARKER_TILE, grid[jnp.clip(chosen, 0, size - 1)])
            )

        # ---- the eye's start and the three eggs ------------------------------ #
        open_mask = grid == SPACE
        open_count = open_mask.sum()
        open_cells = jnp.sort(jnp.where(open_mask, cells, size))
        state, picks = _simple_choose(state, open_count, 1 + TOTAL_ENEMIES)
        chosen = open_cells[jnp.clip(picks, 0, size - 1)]
        start_cell = chosen[0]
        egg_cells = chosen[1:]
        for slot in range(TOTAL_ENEMIES):
            grid = grid.at[jnp.clip(egg_cells[slot], 0, size - 1)].set(MARKER_TILE)

        # ---- every remaining open cell becomes an orb ------------------------ #
        total_orbs = open_count
        grid = jnp.where(open_mask, ORB_TILE, grid)
        grid = jnp.where(grid == MARKER_TILE, SPACE, grid)

        # ---- the bodies ------------------------------------------------------- #
        pool = empty_pool()
        slot = jnp.zeros((), jnp.int32)

        def write(pool, slot, ent, active):
            position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
            pool = jax.tree.map(
                lambda buf, value, s=position, on=active: buf.at[s].set(jnp.where(on, value, buf[s])),
                pool,
                ent,
            )
            return pool, slot + active

        for index in range(NUM_QUADRANTS):
            cell = crystal_cells[index]
            crystal = engine.entity(
                (cell % MAZE_DIM).astype(jnp.float32) + 0.5,
                (cell // MAZE_DIM).astype(jnp.float32) + 0.5,
                rx=ORB_RADIUS,
                ry=ORB_RADIUS,
                type=LARGE_ORB,
                image_type=LARGE_ORB,
            )
            pool, slot = write(pool, slot, crystal, cell < size)

        for index in range(TOTAL_ENEMIES):
            cell = egg_cells[index]
            egg = engine.entity(
                (cell % MAZE_DIM).astype(jnp.float32) + 0.5,
                (cell // MAZE_DIM).astype(jnp.float32) + 0.5,
                rx=ENEMY_RADIUS,
                ry=ENEMY_RADIUS,
                type=ENEMY_EGG,
                image_type=ENEMY_EGG,
                health=EGG_TIMEOUT,
            )
            pool, slot = write(pool, slot, egg, jnp.asarray(True))

        agent = engine.entity(
            (start_cell % MAZE_DIM).astype(jnp.float32) + 0.5,
            (start_cell // MAZE_DIM).astype(jnp.float32) + 0.5,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )

        is_space = grid != MAZE_WALL
        free_list = jnp.sort(jnp.where(is_space, cells, size))
        return ChaserState(
            grid=grid,
            agent=agent,
            entities=pool,
            free_cells=free_list,
            free_count=is_space.sum(),
            is_space=is_space,
            eat_time=jnp.asarray(-EAT_TIMEOUT, jnp.int32),
            total_orbs=total_orbs,
            orbs_collected=jnp.zeros((), jnp.int32),
            total_enemies=jnp.asarray(TOTAL_ENEMIES, jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: ChaserState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1
        grid = env_state.grid
        size = MAZE_DIM * MAZE_DIM

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        # ``update_agent_velocity``: snap to maxspeed and keep the sign.
        agent = env_state.agent
        vx = jnp.where(action_vx != 0, MAX_SPEED * action_vx, agent.vx)
        vy = jnp.where(action_vy != 0, MAX_SPEED * action_vy, agent.vy)
        agent = agent._replace(vx=jnp.sign(vx) * MAX_SPEED, vy=jnp.sign(vy) * MAX_SPEED, is_reflected=action_vx < 0)

        agent, _ = engine.step_object(
            agent,
            grid.reshape(MAZE_DIM, MAZE_DIM),
            out_of_bounds=jnp.asarray(MAZE_WALL),
            is_blocked=lambda tile, is_horizontal: tile == MAZE_WALL,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(env_state.entities)

        eaten_time = env_state.eat_time
        can_eat = (now - eaten_time) < EAT_TIMEOUT
        vscale = jnp.where(can_eat, MAX_SPEED * 0.5, MAX_SPEED)
        dist_scale = jnp.where(can_eat, -1, 1)

        positions = jnp.arange(size, dtype=jnp.int32)
        nbr, _ = engine_neighbours(positions)

        agent_index = (agent.y.astype(jnp.int32) * MAZE_DIM + agent.x.astype(jnp.int32)).clip(0, size - 1)

        def ghost_turn(index, carry):
            pool, eaten_time = carry
            ent = jax.tree.map(lambda field: field[index], pool)
            is_egg = ent.active & (ent.type == ENEMY_EGG)
            hatched = is_egg & (ent.health - 1 <= 0)
            pool = pool._replace(health=jnp.where(is_egg, ent.health - 1, pool.health))
            pool = pool._replace(active=pool.active.at[index].set(pool.active[index] & ~hatched))

            is_ghost = ent.active & (ent.type == ENEMY)
            cell_x = ent.x - 0.5
            cell_y = ent.y - 0.5
            at_junction = jnp.abs(cell_x - jnp.round(cell_x)) + jnp.abs(cell_y - jnp.round(cell_y)) < 0.01
            idle = (ent.vx == 0) & (ent.vy == 0)
            moving = is_ghost & (idle | at_junction)

            ghost_index = jnp.clip((cell_y.astype(jnp.int32) * MAZE_DIM + cell_x.astype(jnp.int32)), 0, size - 1)
            previous = jnp.clip(
                (
                    (cell_y - jnp.sign(ent.vy)).astype(jnp.int32) * MAZE_DIM
                    + (cell_x - jnp.sign(ent.vx)).astype(jnp.int32)
                ),
                0,
                size - 1,
            )
            options = nbr[ghost_index]
            is_open = env_state.is_space[jnp.clip(options, 0, size - 1)]
            usable = is_open & (options != previous) & (options >= 0) & (options < size)

            distance = jnp.abs(options % MAZE_DIM - agent_index % MAZE_DIM) + jnp.abs(
                options // MAZE_DIM - agent_index // MAZE_DIM
            )
            weighted = distance * dist_scale
            best = jnp.where(usable, weighted, 2 * MAZE_DIM).min()
            aggressive = step_rand_int % 2 == 0
            keep = usable & (~aggressive | (weighted == best))

            order = jnp.where(keep, jnp.arange(4), 4)
            ranked = jnp.sort(order)
            count = keep.sum()
            pick = ranked[step_rand_int % jnp.maximum(count, 1)]
            target = options[jnp.clip(pick, 0, 3)]
            next_x = target % MAZE_DIM
            next_y = target // MAZE_DIM

            pool = pool._replace(
                vx=jnp.where(moving & (count > 0), (next_x - cell_x) * vscale, pool.vx),
                vy=jnp.where(moving & (count > 0), (next_y - cell_y) * vscale, pool.vy),
            )

            # a hatched egg becomes a ghost right here
            ghost = engine.entity(
                ent.x,
                ent.y,
                rx=ENEMY_RADIUS,
                ry=ENEMY_RADIUS,
                type=ENEMY,
                image_type=ENEMY,
                smart_step=True,
                active=hatched,
            )
            free = ~pool.active
            slot = jnp.argmax(free)
            pool = jax.lax.cond(
                hatched & free.any(),
                lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, ghost),
                lambda p: p,
                pool,
            )
            return pool, eaten_time

        pool, eaten_time = jax.lax.fori_loop(0, MAX_ENTITIES, ghost_turn, (pool, eaten_time))

        # ---- keep the ghost count up ------------------------------------------ #
        num_enemies = ((pool.type == ENEMY) | (pool.type == ENEMY_EGG)) & pool.active
        respawn = num_enemies.sum() < env_state.total_enemies
        free_cell = env_state.free_cells[jnp.clip(step_rand_int % jnp.maximum(env_state.free_count, 1), 0, size - 1)]
        egg = engine.entity(
            (free_cell % MAZE_DIM).astype(jnp.float32) + 0.5,
            (free_cell // MAZE_DIM).astype(jnp.float32) + 0.5,
            rx=ENEMY_RADIUS,
            ry=ENEMY_RADIUS,
            type=ENEMY_EGG,
            image_type=ENEMY_EGG,
            health=EGG_TIMEOUT,
            active=respawn,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            respawn & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, egg),
            lambda p: p,
            pool,
        )

        # ---- contact with the eye --------------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        ate_crystal = touching & (pool.type == LARGE_ORB)
        eaten_time = jnp.where(ate_crystal.any(), now, eaten_time)
        eaten_ghost = touching & (pool.type == ENEMY) & can_eat
        killed = touching & (pool.type == ENEMY) & ~can_eat
        reward = ORB_REWARD * (ate_crystal.sum() + eaten_ghost.sum()).astype(jnp.float32)
        pool = pool._replace(active=pool.active & ~(ate_crystal | eaten_ghost))

        # ---- the orb under the eye -------------------------------------------- #
        standing = grid[agent_index]
        collected = standing == ORB_TILE
        grid = grid.at[agent_index].set(jnp.where(collected, SPACE, standing))
        orbs_collected = env_state.orbs_collected + collected
        reward = reward + ORB_REWARD * collected.astype(jnp.float32)

        finished = orbs_collected >= env_state.total_orbs
        reward = reward + jnp.where(finished, COMPLETION_BONUS, 0.0)
        terminated = killed.any() | finished

        stepped = ChaserState(
            grid=grid,
            agent=agent,
            entities=pool,
            free_cells=env_state.free_cells,
            free_count=env_state.free_count,
            is_space=env_state.is_space,
            eat_time=eaten_time,
            total_orbs=env_state.total_orbs,
            orbs_collected=orbs_collected,
            total_enemies=env_state.total_enemies,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: ChaserState,
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

    def observe(self, env_state: ChaserState) -> jax.Array:
        """Whole maze on screen: ``center_agent`` is off."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        return render_textured(
            env_state.grid.reshape(MAZE_DIM, MAZE_DIM),
            jnp.asarray([MAZE_DIM / 2, MAZE_DIM / 2]),
            bodies,
            TILE_ROWS,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            MAZE_DIM,
            jnp.asarray(MAZE_WALL),
            background=FLOOR_COLOUR,
        )


def engine_neighbours(positions):
    """The four orthogonal neighbours in the order the C++ scans them, as ``(nbr, ok)``."""
    x = positions % MAZE_DIM
    y = positions // MAZE_DIM
    offsets = jnp.asarray([-1, -MAZE_DIM, MAZE_DIM, 1], jnp.int32)
    nbr = positions[:, None] + offsets[None, :]
    valid = jnp.stack([x > 0, y > 0, y < MAZE_DIM - 1, x < MAZE_DIM - 1], axis=1)
    return jnp.where(valid, nbr, -1), valid
