"""A pure-JAX port of ProcGen's ``bigfish``, from ``procgen/src/games/bigfish.cpp``.

An underwater arena: the agent swims in eight directions, eats any fish smaller
than itself (+1, and it grows), dies on contact with a bigger one, and finishes
after 30 fish (+10).  New fish keep swimming in from the left and right edges.

There is no gravity and no terrain -- the grid is entirely empty and the world is
bounded by ``out_of_bounds_object = INVALID_OBJ``, which blocks movement the same
way a wall would.  The camera is fixed on the whole 20x20 world rather than
following the agent (``options.center_agent = false``), so the observation shows
everything.

Deviations, all deliberate: sprites come from a 32x32 atlas, the water background
is a flat colour, and the eight-direction action grid is decoded as in the
original (slots 9-14 are no-ops, as in the other games here).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.objects import SPACE
from procgen_jax.render import render_textured

# ``bigfish.cpp`` object ids and constants.
FISH = 2
#: ``const int COMPLETION_BONUS = 10.0f;`` -- yes, an int.
COMPLETION_BONUS = 10.0
#: ``const int POSITIVE_REWARD = 1.0f;``
POSITIVE_REWARD = 1.0
FISH_MIN_R = 0.25
FISH_MAX_R = 2.0
FISH_QUOTA = 30

#: ``main_width = main_height = 20``, ``timeout = 6000``.
WORLD_DIM = 20
TIMEOUT = 6000
NUM_ACTIONS = 15
#: ``BasicAbstractGame`` defaults, unchanged by this game.
MIXRATE = 0.5
MAX_SPEED = 0.5
#: ``float start_r = .5;`` (1.0 in EasyMode, which is not implemented).
START_R = 0.5
#: ``BasicAbstractGame::game_reset`` places the agent with ``a_r = 0.4``.
START_MARGIN = 0.4

#: The three fish sprites ``asset_for_type`` builds, in order.
FISH_SPRITES = ['misc/fish074', 'misc/fish078', 'misc/fish080']
#: The agent's sprite.
PLAYER_SPRITE = 'misc/fish072'
#: Spawns are appended to a ring buffer; typical occupancy is under ten.
MAX_FISH = 64
#: ``|v| <= maxspeed * mixrate / (1 - (1 - mixrate) * 0.9) < 0.46``, so 4 is enough.
MAX_SUB_STEPS = 4
#: ``out_of_bounds_object`` is ``INVALID_OBJ`` here, not a wall.
OUT_OF_BOUNDS = -1
#: Water, standing in for ProcGen's background image.
WATER = (60, 120, 190)


def _tile_sprite_table() -> jax.Array:
    """The grid is empty, so every tile is transparent."""
    return jnp.full((64,), sprites.row('_transparent'), jnp.int32)


def _entity_sprite_table() -> jax.Array:
    table = jnp.full((32, 8), sprites.row('_transparent'), jnp.int32)
    table = table.at[engine.PLAYER, 0].set(sprites.row(PLAYER_SPRITE))
    for theme, name in enumerate(FISH_SPRITES):
        table = table.at[FISH, theme].set(sprites.row(name))
    return table


TILE_SPRITES = _tile_sprite_table()
ENTITY_SPRITES = _entity_sprite_table()
FISH_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(name)] for name in FISH_SPRITES])


class BigFishState(NamedTuple):
    """Game state: an empty grid, the agent, and the fish ring buffer."""

    grid: jax.Array  # (20, 20) int32, all SPACE
    agent: engine.Entity
    fish: engine.Entity  # (MAX_FISH,) ring buffer
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937
    fish_eaten: jax.Array
    r_inc: jax.Array
    next_slot: jax.Array


def empty_fish_pool() -> engine.Entity:
    zeros = jnp.zeros((MAX_FISH,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((MAX_FISH,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (MAX_FISH,)), base)


class BigFishEnv(Env):
    """ProcGen's ``bigfish`` as a functional, jit-friendly environment."""

    def __init__(self, timeout: int = TIMEOUT, level_seed: int | None = None):
        self.timeout = timeout
        #: Pin a level seed to replay exactly the level the C++ implementation builds.
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

        def reset(env_state: BigFishState) -> BigFishState:
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
    def reset_env_state(self, key: jax.Array) -> BigFishState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        # BasicAbstractGame::game_reset draws, in order.
        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, agent_x = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y (overridden below)

        x = agent_x * (WORLD_DIM - 2 * START_MARGIN) + START_MARGIN
        start_r = jnp.asarray(START_R, jnp.float32)
        return BigFishState(
            grid=jnp.full((WORLD_DIM, WORLD_DIM), SPACE, jnp.int32),
            agent=engine.entity(
                x,
                1 + start_r,
                rx=start_r,
                ry=start_r,
                type=engine.PLAYER,
                image_type=engine.PLAYER,
                smart_step=True,
            ),
            fish=empty_fish_pool(),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
            fish_eaten=jnp.zeros((), jnp.int32),
            r_inc=jnp.asarray((FISH_MAX_R - START_R) / FISH_QUOTA, jnp.float32),
            next_slot=jnp.zeros((), jnp.int32),
        )

    def step_env_state(self, env_state: BigFishState, action: jax.Array):
        grid = env_state.grid
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)

        # ---- game_step: decode the action, then swim ----------------------- #
        move_action = jnp.where(action >= 9, 4, action % 9)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        agent = env_state.agent
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        # decay_agent_velocity()
        agent = agent._replace(vx=0.9 * vx, vy=0.9 * vy)

        agent, _ = engine.step_object(
            agent,
            grid,
            out_of_bounds=jnp.asarray(OUT_OF_BOUNDS),
            is_blocked=lambda tile, is_horizontal: tile == OUT_OF_BOUNDS,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # ---- step_entities: fish drift, then out-of-bounds ones are erased -- #
        fish = env_state.fish
        fish = fish._replace(x=jnp.where(fish.active, fish.x + fish.vx, fish.x))
        outside = engine.is_out_of_bounds(fish, WORLD_DIM, WORLD_DIM)
        fish = fish._replace(active=fish.active & ~outside)

        # ---- handle_agent_collision, in the original's reverse order ------- #
        def collide(index, carry):
            agent, fish, reward, eaten, killed = carry
            slot = MAX_FISH - 1 - index
            fish_rx = fish.rx[slot]
            hit = engine.collides(jax.tree.map(lambda field: field[slot], fish), agent) & fish.active[slot]
            bigger = hit & (fish_rx > agent.rx)
            ate = hit & (fish_rx <= agent.rx)
            agent = agent._replace(
                rx=agent.rx + jnp.where(ate, env_state.r_inc, 0.0),
                ry=agent.ry + jnp.where(ate, env_state.r_inc, 0.0),
            )
            # The original only erases what it eats; a bigger fish just ends the episode.
            fish = fish._replace(active=fish.active.at[slot].set(fish.active[slot] & ~ate))
            return (
                agent,
                fish,
                reward + jnp.where(ate, POSITIVE_REWARD, 0.0),
                eaten + jnp.where(ate, 1, 0),
                killed | bigger,
            )

        agent, fish, reward, eaten, killed = jax.lax.fori_loop(
            0,
            MAX_FISH,
            collide,
            (agent, fish, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), jnp.asarray(False)),
        )
        fish_eaten = env_state.fish_eaten + eaten

        # ---- spawn: one draw every step, five more when it fires ----------- #
        state, spawn_draw = rng.randn(env_state.rng, 10)
        spawn = spawn_draw == 1

        def no_spawn(state):
            zero = jnp.zeros((), jnp.float32)
            return state, (zero, zero, 0.0, zero, jnp.zeros((), jnp.int32))

        def do_spawn(state):
            state, u1 = rng.rand01(state)  # radius
            state, u2 = rng.rand01(state)  # y
            state, u3 = rng.rand01(state)  # direction
            state, u4 = rng.rand01(state)  # speed
            state, theme = rng.randn(state, len(FISH_SPRITES))
            return state, (u1, u2, u3, u4, theme)

        state, (u1, u2, u3, u4, theme) = jax.lax.cond(spawn, do_spawn, no_spawn, state)

        radius = (FISH_MAX_R - FISH_MIN_R) * jnp.power(u1, 1.4) + FISH_MIN_R
        spawn_y = u2 * (WORLD_DIM - 2 * radius)
        moves_right = u3 < 0.5
        spawn_vx = (0.15 + u4 * 0.25) * jnp.where(moves_right, 1.0, -1.0)
        spawn_x = jnp.where(moves_right, -radius, WORLD_DIM + radius)
        new_fish = engine.entity(
            spawn_x,
            spawn_y,
            spawn_vx,
            0.0,
            rx=radius,
            ry=radius / FISH_ASPECTS[theme],
            type=FISH,
            image_type=FISH,
            image_theme=theme,
            is_reflected=~moves_right,
            active=spawn,
        )
        # The original appends to a vector only when a fish spawns; writing every
        # step would clobber a live fish with an inactive slot.
        slot = jnp.mod(env_state.next_slot, MAX_FISH)
        fish = jax.lax.cond(
            spawn,
            lambda pool: jax.tree.map(lambda buf, value: buf.at[slot].set(value), pool, new_fish),
            lambda pool: pool,
            fish,
        )

        # ---- quota, death, bookkeeping ------------------------------------- #
        quota = fish_eaten >= FISH_QUOTA
        reward = reward + jnp.where(quota, COMPLETION_BONUS, 0.0)
        terminated = killed | quota | engine.is_out_of_bounds(agent, WORLD_DIM, WORLD_DIM)
        facing_right = jnp.where(action_vx > 0, True, jnp.where(action_vx < 0, False, agent.is_reflected))
        agent = agent._replace(is_reflected=~facing_right)

        stepped = BigFishState(
            grid=grid,
            agent=agent,
            fish=fish,
            steps=env_state.steps + 1,
            key=key,
            rng=state,
            fish_eaten=fish_eaten,
            r_inc=env_state.r_inc,
            next_slot=env_state.next_slot + spawn.astype(jnp.int32),
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------ #
    def state_from(
        self,
        env_state: BigFishState,
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

    def observe(self, env_state: BigFishState) -> jax.Array:
        """The whole 20x20 world, as ``center_agent`` is off in the original."""
        bodies = jax.tree.map(
            lambda agent, fish: jnp.concatenate([agent[None], fish]),
            env_state.agent,
            env_state.fish,
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
            background=WATER,
        )
