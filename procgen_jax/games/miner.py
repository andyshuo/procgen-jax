"""A pure-JAX port of ProcGen's ``miner``, from ``procgen/src/games/miner.cpp``.

Boulder Dash: the whole 20x20 world starts as dirt, buried diamonds and boulders are
scattered through it, and the robot digs one tile per action.  Boulders and diamonds
fall into empty space below them, roll sideways off other round objects, and end the
run if they land on the robot.  Every diamond pays one and the exit pays ten, but only
once the last diamond is out.

The original's ``handle_push`` never fires (it asks for ``action_vx == 1`` and
``agent->vx == 0`` at the same time, which ``grid_step`` makes impossible), so it is
not reproduced.

Deviations: sprites are not rotated.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``miner.cpp`` object ids; the grid and the entities share them.
BOULDER = 1
DIAMOND = 2
MOVING_BOULDER = 3
MOVING_DIAMOND = 4
ENEMY = 5
EXIT = 6
DIRT = 9
OOB_WALL = 10
#: ProcGen's ``SPACE``.
SPACE = 100

COMPLETION_BONUS = 10.0
DIAMOND_REWARD = 1.0
WORLD_DIM = 20
TIMEOUT = 1000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.5
VISIBILITY = WORLD_DIM
#: ``diamond_pct = 12 / 400`` and ``boulder_pct = 80 / 400`` of the 400 cells.
NUM_DIAMONDS = 12
NUM_BOULDERS = 80
AGENT_RADIUS = 0.5
EXIT_RADIUS = 0.5
MAX_ENTITIES = 16
MAX_SUB_STEPS = 6

TILE_ROWS = (
    jnp.full((128,), sprites.row('_transparent'), jnp.int32)
    .at[BOULDER]
    .set(sprites.row('misc/miner/elementStone007'))
    .at[DIAMOND]
    .set(sprites.row('misc/miner/gemBlue'))
    .at[DIRT]
    .set(sprites.row('misc/miner/dirt'))
    .at[OOB_WALL]
    .set(sprites.row('misc/miner/tile_bricksGrey'))
)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/miner/robot_greenDrive1'))
    .at[EXIT, 0]
    .set(sprites.row('misc/miner/window'))
)
#: The original paints a platformer background; this is a flat stand-in.
CAVE_COLOUR = (40, 30, 22)


class MinerState(NamedTuple):
    """Game state: the dug-out cave, the robot, the exit and the diamond count."""

    grid: jax.Array  # (WORLD_DIM * WORLD_DIM,) DIRT, DIAMOND, BOULDER, SPACE, ...
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) the exit
    diamonds_remaining: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def _simple_choose(state, size, count):
    chosen = jnp.zeros((count,), jnp.int32)
    taken = jnp.zeros((WORLD_DIM * WORLD_DIM,), bool)
    limit = WORLD_DIM * WORLD_DIM - 1
    for pick in range(count):

        def body(carry, taken_at=taken):
            state, _, _ = carry
            state, draw = rng.randn(state, size)
            return state, draw, taken_at[jnp.clip(draw, 0, limit)]

        state, value, _ = jax.lax.while_loop(
            lambda carry: carry[2], body, (state, jnp.zeros((), jnp.int32), jnp.asarray(True))
        )
        chosen = chosen.at[pick].set(value)
        taken = taken.at[jnp.clip(value, 0, limit)].set(True)
    return state, chosen


class MinerEnv(Env):
    """ProcGen's ``miner`` as a functional, jit-friendly environment."""

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

        def reset(env_state: MinerState) -> MinerState:
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
    def reset_env_state(self, key: jax.Array) -> MinerState:
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
        state, picks = _simple_choose(state, size, NUM_DIAMONDS + NUM_BOULDERS + 1)
        start = picks[0]
        agent_x = start % WORLD_DIM
        agent_y = start // WORLD_DIM

        flat = jnp.full((size,), DIRT, jnp.int32)
        diamond_cells = picks[1 : 1 + NUM_DIAMONDS]
        boulder_cells = picks[1 + NUM_DIAMONDS :]
        flat = flat.at[diamond_cells].set(DIAMOND)
        flat = flat.at[boulder_cells].set(BOULDER)

        # the dirt list is taken before the robot's cell and its square are cleared
        dirt_mask = flat == DIRT

        flat = flat.at[agent_y * WORLD_DIM + agent_x].set(SPACE)

        # a boulder next to the robot is buried, so it cannot start squashed
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                ox = agent_x + i
                oy = agent_y + j
                inside = (ox >= 0) & (ox < WORLD_DIM) & (oy >= 0) & (oy < WORLD_DIM)
                cell = jnp.clip(oy * WORLD_DIM + ox, 0, size - 1)
                buried = inside & (flat[cell] == BOULDER)
                flat = flat.at[cell].set(jnp.where(buried, DIRT, flat[cell]))

        # the exit sits where the tile above is dirt or off the top
        above = jnp.clip(cells + WORLD_DIM, 0, size - 1)
        above_solid = (flat[above] == DIRT) | (cells + WORLD_DIM >= size)
        is_eligible = dirt_mask & above_solid
        eligible = jnp.sort(jnp.where(is_eligible, cells, size))
        eligible_count = is_eligible.sum()
        state, exit_pick = rng.randn(state, jnp.maximum(eligible_count, 1))
        exit_cell = eligible[jnp.clip(exit_pick, 0, size - 1)]
        flat = flat.at[exit_cell].set(SPACE)

        pool = empty_pool()
        exit_entity = engine.entity(
            (exit_cell % WORLD_DIM).astype(jnp.float32) + 0.5,
            (exit_cell // WORLD_DIM).astype(jnp.float32) + 0.5,
            rx=EXIT_RADIUS,
            ry=EXIT_RADIUS,
            type=EXIT,
            image_type=EXIT,
            render_z=-1,
            active=True,
        )
        pool = jax.tree.map(lambda buf, value: buf.at[0].set(value), pool, exit_entity)

        agent = engine.entity(
            agent_x.astype(jnp.float32) + 0.5,
            agent_y.astype(jnp.float32) + 0.5,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )
        return MinerState(
            grid=flat,
            agent=agent,
            entities=pool,
            diamonds_remaining=jnp.asarray(NUM_DIAMONDS, jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: MinerState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        grid = env_state.grid
        agent = env_state.agent

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = jnp.where(action_vx != 0, 0, move_action % 3 - 1)

        # ``grid_step``: the velocity *is* the action, one tile per step
        agent = agent._replace(
            vx=action_vx.astype(jnp.float32),
            vy=action_vy.astype(jnp.float32),
            is_reflected=jnp.where(action_vx > 0, False, jnp.where(action_vx < 0, True, agent.is_reflected)),
        )
        view = grid.reshape(WORLD_DIM, WORLD_DIM)
        agent, _ = engine.step_object(
            agent,
            view,
            out_of_bounds=jnp.asarray(OOB_WALL),
            is_blocked=lambda tile, is_horizontal: (tile == BOULDER) | (tile == MOVING_BOULDER),
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # ---- dig -------------------------------------------------------------- #
        agent_index = jnp.clip(agent.y.astype(jnp.int32) * WORLD_DIM + agent.x.astype(jnp.int32), 0, grid.shape[0] - 1)
        under = grid[agent_index]
        reward = jnp.where(under == DIAMOND, DIAMOND_REWARD, 0.0)
        dug = (under == DIRT) | (under == DIAMOND)
        grid = grid.at[agent_index].set(jnp.where(dug, SPACE, under))

        # ---- gravity, one pass from the bottom up ----------------------------- #
        size = grid.shape[0]
        agent_cell = agent_index

        def fall(index, carry):
            grid, killed = carry
            obj = grid[index]
            round_obj = (obj == BOULDER) | (obj == MOVING_BOULDER) | (obj == DIAMOND) | (obj == MOVING_DIAMOND)
            column = index % WORLD_DIM
            below = jnp.clip(index - WORLD_DIM, 0, size - 1)
            has_below = index >= WORLD_DIM
            below_obj = grid[below]
            stationary = jnp.where(
                (obj == MOVING_BOULDER) | (obj == MOVING_DIAMOND),
                jnp.where(obj == MOVING_DIAMOND, DIAMOND, BOULDER),
                obj,
            )
            moving = jnp.where(obj == MOVING_DIAMOND, MOVING_DIAMOND, MOVING_BOULDER)

            empty_below = round_obj & has_below & (below_obj == SPACE) & (agent_cell != below)
            squash = round_obj & (agent_cell == below) & ((obj == MOVING_BOULDER) | (obj == MOVING_DIAMOND))
            # roll off another round object when both the side and the cell above it are free
            round_below = (
                (below_obj == BOULDER)
                | (below_obj == MOVING_BOULDER)
                | (below_obj == DIAMOND)
                | (below_obj == MOVING_DIAMOND)
            )
            left = jnp.clip(index - 1, 0, size - 1)
            right = jnp.clip(index + 1, 0, size - 1)
            left_free = (grid[left] == SPACE) & (grid[jnp.clip(left - WORLD_DIM, 0, size - 1)] == SPACE)
            right_free = (grid[right] == SPACE) & (grid[jnp.clip(right - WORLD_DIM, 0, size - 1)] == SPACE)
            roll_left = round_obj & round_below & (column > 0) & left_free & (agent_cell != left)
            roll_right = round_obj & round_below & (column < WORLD_DIM - 1) & right_free & (agent_cell != right)

            grid = jnp.where(empty_below, grid.at[below].set(moving), grid)
            grid = jnp.where(empty_below, grid.at[index].set(SPACE), grid)
            grid = jnp.where(roll_left, grid.at[left].set(stationary), grid)
            grid = jnp.where(roll_left, grid.at[index].set(SPACE), grid)
            grid = jnp.where(roll_right, grid.at[right].set(stationary), grid)
            grid = jnp.where(roll_right, grid.at[index].set(SPACE), grid)
            # anything that could not move settles back to its stationary type
            stayed = round_obj & ~empty_below & ~roll_left & ~roll_right
            grid = jnp.where(stayed, grid.at[index].set(stationary), grid)
            return grid, killed | squash

        grid, squashed = jax.lax.fori_loop(0, size, fall, (grid, jnp.asarray(False)))

        diamonds_remaining = ((grid == DIAMOND) | (grid == MOVING_DIAMOND)).sum()

        # ---- the exit ---------------------------------------------------------- #
        pool = jax.vmap(engine.entity_step)(env_state.entities)
        touching = engine.collides(agent, pool) & pool.active
        reached_exit = (touching & (pool.type == EXIT)).any() & (diamonds_remaining == 0)
        reward = reward + jnp.where(reached_exit, COMPLETION_BONUS, 0.0)
        terminated = reached_exit | squashed

        stepped = MinerState(
            grid=grid,
            agent=agent,
            entities=pool,
            diamonds_remaining=diamonds_remaining,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: MinerState,
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

    def observe(self, env_state: MinerState) -> jax.Array:
        """``center_agent`` is off, so the whole 20x20 mine is on screen."""
        bodies = jax.tree.map(
            lambda agent, pool: jnp.concatenate([agent[None], pool]),
            env_state.agent,
            env_state.entities,
        )
        return render_textured(
            env_state.grid.reshape(WORLD_DIM, WORLD_DIM),
            jnp.asarray([WORLD_DIM / 2, WORLD_DIM / 2]),
            bodies,
            TILE_ROWS,
            ENTITY_SPRITES[bodies.image_type, bodies.image_theme],
            jnp.ones_like(bodies.rx),
            sprites.PIXELS,
            VISIBILITY,
            jnp.asarray(OOB_WALL),
            background=CAVE_COLOUR,
        )
