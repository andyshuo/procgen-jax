"""A pure-JAX port of ProcGen's ``heist``, from ``procgen/src/games/heist.cpp``.

A locked-maze heist: find the gem behind locked doors, collecting the matching key
for each door on the way.  The level is a maze with doors punched into its forks
(:mod:`procgen_jax.mazegen`), placed somewhere inside an otherwise solid world.

Deviations: the key ring along the top of the original's screen is not drawn (it is
a HUD element in absolute coordinates), sprites are not rotated, and the maze is
copied without the original's per-type sprite themes beyond the three key/lock
colours.

``timeout`` is ProcGen's ``Game`` default of 1000 steps: ``heist.cpp`` never sets
it and neither ``vecgame.cpp`` nor ``builder.py`` overrides it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, mazegen, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``heist.cpp`` object ids.
LOCKED_DOOR = 1
KEY = 2
EXIT = 9
#: ``COMPLETION_BONUS = 10.0f``, assigned (not added) to the reward.
COMPLETION_BONUS = 10.0

#: ``choose_world_dim`` in hard mode; the camera shows all of it.
WORLD_DIM = 13
TIMEOUT = 1000
NUM_ACTIONS = 15
#: ``BasicAbstractGame`` defaults plus ``maxspeed = .75`` from ``choose_world_dim``.
MIXRATE = 0.5
MAX_SPEED = 0.75
MIN_MAZE_DIM = 5
MAX_KEYS = 3
#: ``agent->rx = .375 * maze_scale`` and ``r_ent = maze_scale / 2``.
AGENT_RADIUS = 0.375
DOOR_RADIUS = 0.5
#: Keys, doors and the gem: seven bodies at most.
MAX_ENTITIES = 2 * MAX_KEYS + 1
MAX_SUB_STEPS = 6
#: ``reposition`` retries until it finds a free spot.
REPOSITION_ATTEMPTS = 100

#: The world grid uses this repo's tile ids: 0 is open floor, 1 is a wall.
WALL_TILE = 1
#: ProcGen's own ids, as offered by :mod:`procgen_jax.mazegen`.
MAZE_WALL = 51
MAZE_EXIT = 52
MAZE_AGENT = 53
MAZE_DOOR = 200
MAZE_KEY = 300

KEY_SPRITES = ['keyBlue', 'keyGreen', 'keyRed']
LOCK_SPRITES = ['lock_blue', 'lock_green', 'lock_red']

TILE_SPRITES = (
    jnp.full((64,), sprites.row('_transparent'), jnp.int32).at[WALL_TILE].set(sprites.row('Ground/Dirt/center'))
)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    .at[EXIT, 0]
    .set(sprites.row('misc/heist/gemYellow'))
    .at[engine.PLAYER, 0]
    .set(sprites.row('misc/heist/spaceAstronauts_008'))
)
for _theme, _name in enumerate(KEY_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[KEY, _theme].set(sprites.row(f'misc/heist/{_name}'))
for _theme, _name in enumerate(LOCK_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[LOCKED_DOOR, _theme].set(sprites.row(f'misc/heist/{_name}'))

KEY_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/heist/{name}')] for name in KEY_SPRITES])
#: The original's top-down background is an image; this is a flat stand-in.
FLOOR_COLOUR = (38, 34, 44)


class HeistState(NamedTuple):
    """Game state: the world grid, the agent, the key ring and the maze bodies."""

    grid: jax.Array  # (WORLD_DIM, WORLD_DIM) int32: 0 = floor, 1 = wall
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) keys, doors and the gem
    has_keys: jax.Array  # (MAX_KEYS,) bool, indexed by door/key theme
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


class HeistEnv(Env):
    """ProcGen's ``heist`` as a functional, jit-friendly environment."""

    def __init__(
        self,
        timeout: int = TIMEOUT,
        level_seed: int | None = None,
        maze_dim: int = 9,
    ):
        if maze_dim < MIN_MAZE_DIM or maze_dim % 2 == 0 or maze_dim > WORLD_DIM:
            raise ValueError(f'maze_dim must be odd, >= {MIN_MAZE_DIM} and <= {WORLD_DIM}')
        self.timeout = timeout
        self.level_seed = level_seed
        #: The original draws this per episode; the generator's wall list depends on
        #: the size, so it is fixed here the same way ``MazeEnv(world_dim=...)`` is.
        self.maze_dim = maze_dim

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

        def reset(env_state: HeistState) -> HeistState:
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
        """``rand_pos``: uniform in ``[low+radius, high-radius]``, or the midpoint."""
        draws = (high - low) > 2 * radius
        state, draw = jax.lax.cond(
            draws & active,
            lambda s: rng.rand01(s),
            lambda s: (s, jnp.zeros((), jnp.float32)),
            state,
        )
        return state, jnp.where(draws, (high - low - 2 * radius) * draw + radius + low, (high + low) / 2)

    def _spawn(self, state, pool, agent, active, low_x, low_y, radius):
        """``spawn_entity``: reposition inside the cell box until the body is clear.

        A bounded ``while_loop`` rather than a ``fori_loop`` with gated draws: an
        inactive slot never enters the loop, so it consumes no random numbers, and
        the body has no nested ``cond`` to compile.
        """

        def cond(carry):
            _, _, _, _, done, attempts = carry
            return jnp.logical_and(jnp.logical_not(done), attempts < REPOSITION_ATTEMPTS)

        def body(carry):
            state, x, y, placed, _, attempts = carry
            state, nx = rng.rand01(state)
            state, ny = rng.rand01(state)
            px = (1.0 - 2 * radius) * nx + radius + low_x
            py = (1.0 - 2 * radius) * ny + radius + low_y
            candidate = engine.entity(px, py, rx=radius, ry=radius)
            blocked = (engine.collides(candidate, pool) & pool.active).any() | engine.collides(candidate, agent)
            fresh = ~blocked
            return (
                state,
                jnp.where(fresh, px, x),
                jnp.where(fresh, py, y),
                placed | fresh,
                fresh,
                attempts + 1,
            )

        start = (
            state,
            jnp.zeros((), jnp.float32),
            jnp.zeros((), jnp.float32),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
        )
        state, x, y, placed, _, _ = jax.lax.while_loop(lambda carry: cond(carry) & active, body, start)
        return state, x, y

    def reset_env_state(self, key: jax.Array) -> HeistState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        # BasicAbstractGame::game_reset draws, in order.
        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        # ``difficulty`` only feeds ``num_keys``: maze_dim is fixed by the constructor.
        state, difficulty = rng.randn(state, (WORLD_DIM - MIN_MAZE_DIM) // 2 + 1)
        state, extra_keys = rng.randn(state, 2)
        num_keys = jnp.minimum(difficulty + extra_keys, MAX_KEYS)

        maze_dim = self.maze_dim
        state, maze = mazegen.generate_maze_with_doors(state, maze_dim, num_keys)
        state, offset_x = rng.randn(state, WORLD_DIM - maze_dim + 1)
        state, offset_y = rng.randn(state, WORLD_DIM - maze_dim + 1)

        # One scatter paints the whole maze; the cells map one to one because
        # maze_scale is 1.
        grid = jnp.full((WORLD_DIM, WORLD_DIM), WALL_TILE, jnp.int32)
        cells = jnp.arange(maze_dim)
        # Full 2-D coordinate grids: the flat lookups below index them with a single
        # (i * maze_dim + j) index, so they must be (maze_dim, maze_dim).
        cell_x = jnp.broadcast_to(offset_x + cells[None, :], (maze_dim, maze_dim))
        cell_y = jnp.broadcast_to(offset_y + cells[:, None], (maze_dim, maze_dim))
        objects = maze[1 : maze_dim + 1, 1 : maze_dim + 1]
        grid = grid.at[cell_y, cell_x].set(jnp.where(objects == MAZE_WALL, WALL_TILE, SPACE))

        is_key = objects >= MAZE_KEY
        is_door = (objects >= MAZE_DOOR) & ~is_key
        is_exit = objects == MAZE_EXIT
        any_body = (is_key | is_door | is_exit).reshape(-1)
        flat_objects = objects.reshape(-1)
        world_x = cell_x.reshape(-1) + 0.5
        world_y = cell_y.reshape(-1) + 0.5

        body_kind = jnp.full((MAX_ENTITIES,), EXIT, jnp.int32)
        body_theme = jnp.zeros((MAX_ENTITIES,), jnp.int32)
        body_cell = jnp.zeros((MAX_ENTITIES, 2), jnp.float32)
        body_active = jnp.zeros((MAX_ENTITIES,), bool)
        rank = jnp.cumsum(any_body.astype(jnp.int32))
        total_bodies = rank[-1]
        for slot in range(MAX_ENTITIES):
            index = jnp.argmax(rank == slot + 1)
            present = slot < total_bodies
            obj = flat_objects[index]
            # NB: not ``key`` -- that is the PRNG key this state carries.
            is_key = obj >= MAZE_KEY
            is_door = (obj >= MAZE_DOOR) & ~is_key
            body_kind = body_kind.at[slot].set(jnp.where(is_key, KEY, jnp.where(is_door, LOCKED_DOOR, EXIT)))
            body_theme = body_theme.at[slot].set(
                jnp.clip(
                    jnp.where(is_key, obj - MAZE_KEY - 1, jnp.where(is_door, obj - MAZE_DOOR - 1, 0)), 0, MAX_KEYS - 1
                )
            )
            body_cell = body_cell.at[slot].set(jnp.stack([world_x[index], world_y[index]]))
            body_active = body_active.at[slot].set(present)

        agent_index = jnp.argmax((objects == MAZE_AGENT).reshape(-1))
        agent_centre = jnp.stack([world_x[agent_index], world_y[agent_index]])

        pool = empty_pool()
        agent_entity = engine.entity(agent_centre[0], agent_centre[1], rx=AGENT_RADIUS, ry=AGENT_RADIUS)
        for slot in range(MAX_ENTITIES):
            kind = body_kind[slot]
            theme = body_theme[slot]
            active = body_active[slot]
            is_door = kind == LOCKED_DOOR
            radius = jnp.where(is_door, DOOR_RADIUS, AGENT_RADIUS)
            aspect = jnp.where(kind == KEY, KEY_ASPECTS[theme], 1.0)
            low_x = body_cell[slot, 0] - 0.5
            low_y = body_cell[slot, 1] - 0.5

            state, spawn_x, spawn_y = self._spawn(state, pool, agent_entity, active & ~is_door, low_x, low_y, radius)
            body = engine.entity(
                jnp.where(is_door, body_cell[slot, 0], spawn_x),
                jnp.where(is_door, body_cell[slot, 1], spawn_y),
                rx=radius,
                ry=radius / aspect,
                type=kind,
                image_type=kind,
                image_theme=theme,
                active=active,
            )
            pool = jax.tree.map(
                lambda buf, value, slot=slot: buf.at[slot].set(value),
                pool,
                body,
            )

        agent = engine.entity(
            agent_centre[0],
            agent_centre[1],
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS,
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            render_z=1,
            smart_step=True,
        )
        return HeistState(
            grid=grid,
            agent=agent,
            entities=pool,
            has_keys=jnp.zeros((MAX_KEYS,), bool),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: HeistState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)

        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy, is_reflected=action_vx < 0)

        pool = env_state.entities

        def doors_block(ent, is_horizontal):
            """``is_blocked_ents``: a door blocks until its key is on the ring."""
            open_doors = engine.collides(ent, pool) & pool.active & (pool.type == LOCKED_DOOR)
            locked = open_doors & ~env_state.has_keys[pool.image_theme]
            return locked.any(), jnp.asarray(False)

        agent, _ = engine.step_object(
            agent,
            env_state.grid,
            out_of_bounds=jnp.asarray(WALL_TILE),
            is_blocked=lambda tile, is_horizontal: tile == WALL_TILE,
            will_reflect=lambda tile: jnp.asarray(False),
            blocked_by_entities=doors_block,
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        pool = jax.vmap(engine.entity_step)(pool)

        # ``push_obj``: a locked door displaces the agent back to flush contact and
        # zeroes its velocity on the axis of travel. The engine only reports the
        # block (coinrun treats crates as rigid), so resolve it here.
        locked_doors = (
            engine.collides(agent, pool)
            & pool.active
            & (pool.type == LOCKED_DOOR)
            & ~env_state.has_keys[pool.image_theme]
        )
        nearest = jnp.argmin(jnp.where(locked_doors, jnp.abs(pool.x - agent.x) + jnp.abs(pool.y - agent.y), 1e9))
        blocked = locked_doors.any()
        door = jax.tree.map(lambda field: field[nearest], pool)
        horizontal = jnp.abs(agent.vx) >= jnp.abs(agent.vy)
        pushed_x = door.x + jnp.sign(agent.x - door.x + 1e-6) * (door.rx + agent.rx)
        pushed_y = door.y + jnp.sign(agent.y - door.y + 1e-6) * (door.ry + agent.ry)
        agent = agent._replace(
            x=jnp.where(blocked & horizontal, pushed_x, agent.x),
            y=jnp.where(blocked & ~horizontal, pushed_y, agent.y),
            vx=jnp.where(blocked & horizontal, 0.0, agent.vx),
            vy=jnp.where(blocked & ~horizontal, 0.0, agent.vy),
        )

        # ---- handle_agent_collision ---------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        picks_up = touching & (pool.type == KEY)
        collected = jnp.zeros((MAX_KEYS,), bool).at[pool.image_theme].set(picks_up, mode='drop')
        has_keys = env_state.has_keys | collected

        reached_gem = (touching & (pool.type == EXIT)).any()
        unlocked = touching & (pool.type == LOCKED_DOOR) & has_keys[pool.image_theme]
        pool = pool._replace(active=pool.active & ~picks_up & ~unlocked)

        terminated = reached_gem | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)
        reward = jnp.where(reached_gem, COMPLETION_BONUS, 0.0)

        stepped = HeistState(
            grid=env_state.grid,
            agent=agent,
            entities=pool,
            has_keys=has_keys,
            steps=env_state.steps + 1,
            key=key,
            rng=env_state.rng,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: HeistState,
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

    def observe(self, env_state: HeistState) -> jax.Array:
        """The whole world: ``center_agent`` is off, so all 13 tiles are visible."""
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
