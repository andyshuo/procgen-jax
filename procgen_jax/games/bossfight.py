"""A pure-JAX port of ProcGen's ``bossfight``, from ``procgen/src/games/bossfight.cpp``.

A bullet-hell boss fight: the boss drifts around the upper part of a 20x20 arena
behind a shield that opens and closes on a timer, meteors sit in the lower half, and
the player's lasers only count while the shield is down.  The boss has several rounds
of health; each round it picks a new spot and switches to another of four patterns.

Deviations: explosion, laser-trail and reflective particles are decoration and are not
spawned, and sprites are not rotated.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from procgen_jax import engine, rng, sprites
from procgen_jax.base import Env, State, auto_reset
from procgen_jax.render import render_textured

# ``bossfight.cpp`` object ids.
PLAYER_BULLET = 1
BOSS = 2
SHIELDS = 3
ENEMY_BULLET = 4
LASER_TRAIL = 5
REFLECTED_BULLET = 6
BARRIER = 7
#: ProcGen's ``SPACE``; the arena has no tiles.
SPACE = 100
#: A sentinel tile for "outside the arena", so the edge blocks the ships the way
#: ``out_of_bounds_object`` does in the original.
OUT_OF_BOUNDS = 101

COMPLETION_BONUS = 10.0
POSITIVE_REWARD = 1.0
BOSS_RADIUS = 3.0
BARRIER_RADIUS = 0.6
BARRIER_HEALTH = 3.0
NUM_ATTACK_MODES = 4
NUM_LASER_THEMES = 3
PLAYER_BULLET_VEL = 1.0
AGENT_RADIUS = 0.75
BULLET_RADIUS = 0.25
BOSS_BULLET_RADIUS = 0.5
BOSS_BULLET_EXPIRE = 50
PLAYER_BULLET_EXPIRE = 25
FIRE_COOLDOWN = 3
BOTTOM_MARGIN = 6
BOSS_VEL_TIMEOUT = 20
BOSS_DAMAGED_TIMEOUT = 40
VULNERABLE_DURATION = 500
WORLD_DIM = 20
TIMEOUT = 4000
NUM_ACTIONS = 15
MIXRATE = 0.5
MAX_SPEED = 0.85
BOSS_BULLET_VEL = 0.75
BASE_FIRE_PROB = 0.1
MAX_ROUNDS = 5
MAX_ROUND_HEALTH = 9
MAX_BARRIERS = 3
#: The widest pattern fires eight bullets in one step.
MAX_SHOTS = 8
MAX_ENTITIES = 96
MAX_SUB_STEPS = 6
BARRIER_THEMES = 8

PLAYER_SPRITES = ['playerShip1_blue', 'playerShip1_green', 'playerShip2_orange', 'playerShip3_red']
BOSS_SPRITES = ['enemyShipBlack1', 'enemyShipBlue2', 'enemyShipGreen3', 'enemyShipRed4']
LASER_SPRITES = ['laserGreen14', 'laserRed11', 'laserBlue09']
BARRIER_SPRITES = [f'spaceMeteors_{index:03d}' for index in range(1, 5)] + [
    f'meteorGrey_big{index}' for index in range(1, 5)
]

TILE_ROWS = jnp.full((128,), sprites.row('_transparent'), jnp.int32)
ENTITY_SPRITES = (
    jnp.full((32, 8), sprites.row('_transparent'), jnp.int32).at[SHIELDS, 0].set(sprites.row('misc/bossfight/shield2'))
)
for _theme, _name in enumerate(PLAYER_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[engine.PLAYER, _theme].set(sprites.row(f'misc/bossfight/{_name}'))
for _theme, _name in enumerate(BOSS_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[BOSS, _theme].set(sprites.row(f'misc/bossfight/{_name}'))
for _theme, _name in enumerate(LASER_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[PLAYER_BULLET, _theme].set(sprites.row(f'misc/bossfight/{_name}'))
    ENTITY_SPRITES = ENTITY_SPRITES.at[ENEMY_BULLET, _theme].set(sprites.row(f'misc/bossfight/{_name}'))
for _theme, _name in enumerate(BARRIER_SPRITES):
    ENTITY_SPRITES = ENTITY_SPRITES.at[BARRIER, _theme].set(sprites.row(f'misc/starpilot/{_name}'))

PLAYER_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/bossfight/{name}')] for name in PLAYER_SPRITES])
BOSS_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/bossfight/{name}')] for name in BOSS_SPRITES])
BARRIER_ASPECTS = jnp.asarray([sprites.ASPECTS[sprites.row(f'misc/starpilot/{name}')] for name in BARRIER_SPRITES])
#: The original paints a space background; this is a flat stand-in.
SPACE_COLOUR = (8, 10, 24)


class BossfightState(NamedTuple):
    """Game state: the boss, its shield, the meteors and the bullets in the air."""

    grid: jax.Array  # (WORLD_DIM * WORLD_DIM,) all SPACE
    agent: engine.Entity
    entities: engine.Entity  # (MAX_ENTITIES,) boss, shield, meteors, bullets
    shields_are_up: jax.Array
    health: jax.Array
    round_num: jax.Array
    round_health: jax.Array
    num_rounds: jax.Array
    attack_modes: jax.Array  # (MAX_ROUNDS,) padded with zeros
    attack_mode: jax.Array
    invulnerable_duration: jax.Array
    time_to_swap: jax.Array
    curr_vel_timeout: jax.Array
    damaged_until_time: jax.Array
    last_fire_time: jax.Array
    player_laser_theme: jax.Array
    boss_laser_theme: jax.Array
    boss_slot: jax.Array
    shield_slot: jax.Array
    steps: jax.Array
    key: jax.Array
    rng: rng.Mt19937


def empty_pool(count: int = MAX_ENTITIES) -> engine.Entity:
    zeros = jnp.zeros((count,), jnp.float32)
    base = engine.entity(zeros, zeros, active=jnp.zeros((count,), bool))
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count,)), base)


def write_slot(pool, slot, ent, live):
    """Write one entity into a slot when ``live``."""
    position = jnp.clip(slot, 0, MAX_ENTITIES - 1)
    return jax.tree.map(
        lambda buf, value, s=position, on=live: buf.at[s].set(jnp.where(on, value, buf[s])),
        pool,
        ent,
    )


def attack_shots(mode, now, rand_pct):
    """``(angles, active)`` for the boss's four patterns at this step.

    ``mode`` is traced, so every pattern is computed and the active one is selected.
    """
    shots = jnp.arange(MAX_SHOTS, dtype=jnp.int32).astype(jnp.float32)
    zero = jnp.zeros((MAX_SHOTS,), jnp.float32)

    # mode 0: five bullets fanned toward the player every eighth step
    fan = 1.5 * jnp.pi + (shots - 2.0) * (jnp.pi / 8)
    mode0 = (now % 8 == 0) & (shots < 5)

    # mode 1: four bullets in a rotating cross every fifth step
    k = jnp.abs(8 - ((now // 5) % 16)).astype(jnp.float32)
    cross = jnp.pi * (1.25 + 0.5 * k / 8.0) + shots * (jnp.pi / 2)
    mode1 = (now % 5 == 0) & (shots < 4)

    # mode 2: an eight-way ring every tenth step, rotated by a random offset
    ring = (2 * jnp.pi / 8) * shots + rand_pct * 2 * jnp.pi
    mode2 = now % 10 == 0

    # mode 3: one aimed shot every fourth step
    aimed = jnp.full((MAX_SHOTS,), jnp.pi * (1 + rand_pct), jnp.float32)
    mode3 = (now % 4 == 0) & (shots < 1)

    angles = jnp.where(
        mode == 0,
        jnp.where(mode0, fan, zero),
        jnp.where(
            mode == 1,
            jnp.where(mode1, cross, zero),
            jnp.where(mode == 2, jnp.where(mode2, ring, zero), jnp.where(mode3, aimed, zero)),
        ),
    )
    active = jnp.where(
        mode == 0,
        mode0,
        jnp.where(mode == 1, mode1, jnp.where(mode == 2, mode2, mode3)),
    )
    return angles, active


class BossfightEnv(Env):
    """ProcGen's ``bossfight`` as a functional, jit-friendly environment."""

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

        def reset(env_state: BossfightState) -> BossfightState:
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
    def reset_env_state(self, key: jax.Array) -> BossfightState:
        if self.level_seed is None:
            level_seed = jax.random.randint(key, (), 0, 2**31 - 1)
        else:
            level_seed = jnp.asarray(self.level_seed, jnp.int32)
        state = rng.seed(level_seed.astype(jnp.uint32))

        state, _ = rng.rand01(state)  # bg_pct_x
        state, _ = rng.randn(state, 8)  # background_index
        state, _ = rng.rand01(state)  # random agent start, x
        state, _ = rng.rand01(state)  # random agent start, y

        state, boss_theme = rng.randn(state, len(BOSS_SPRITES))
        state, round_health = rng.randn(state, MAX_ROUND_HEALTH)
        round_health = round_health + 1
        state, num_rounds = rng.randn(state, MAX_ROUNDS)
        num_rounds = num_rounds + 1
        state, invulnerable = rng.randn(state, 4)
        invulnerable = invulnerable + 2
        state, player_theme = rng.randn(state, len(PLAYER_SPRITES))
        state, player_laser_theme = rng.randn(state, NUM_LASER_THEMES)
        state, boss_laser_theme = rng.randn(state, NUM_LASER_THEMES)

        attack_modes = jnp.zeros((MAX_ROUNDS,), jnp.int32)
        for index in range(MAX_ROUNDS):
            live = index < num_rounds
            state, mode = jax.lax.cond(
                live,
                lambda st: rng.randn(st, NUM_ATTACK_MODES),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            attack_modes = attack_modes.at[index].set(jnp.where(live, mode, attack_modes[index]))

        pool = empty_pool()
        boss = engine.entity(
            WORLD_DIM / 2,
            WORLD_DIM / 2,
            rx=BOSS_RADIUS,
            ry=BOSS_RADIUS / BOSS_ASPECTS[jnp.clip(boss_theme, 0, len(BOSS_SPRITES) - 1)],
            type=BOSS,
            image_type=BOSS,
            image_theme=boss_theme,
            active=True,
        )
        shields = engine.entity(
            boss.x,
            boss.y,
            rx=1.2 * boss.rx,
            ry=1.2 * boss.ry,
            type=SHIELDS,
            image_type=SHIELDS,
            active=True,
        )
        pool = write_slot(pool, jnp.asarray(0, jnp.int32), boss, jnp.asarray(True))
        pool = write_slot(pool, jnp.asarray(1, jnp.int32), shields, jnp.asarray(True))

        agent = engine.entity(
            WORLD_DIM / 2,
            AGENT_RADIUS,
            rx=AGENT_RADIUS,
            ry=AGENT_RADIUS / PLAYER_ASPECTS[jnp.clip(player_theme, 0, len(PLAYER_SPRITES) - 1)],
            type=engine.PLAYER,
            image_type=engine.PLAYER,
            image_theme=player_theme,
            render_z=1,
            smart_step=True,
        )

        state, _ = rng.rand01(state)  # barriers_moves_right
        state, barrier_draw = rng.randn(state, 3)
        barrier_count = barrier_draw + 1

        slot = jnp.asarray(2, jnp.int32)
        for index in range(MAX_BARRIERS):
            spawn = index < barrier_count
            state, y_draw = jax.lax.cond(
                spawn, lambda st: rng.rand01(st), lambda st: (st, jnp.zeros((), jnp.float32)), state
            )
            state, x_draw = jax.lax.cond(
                spawn, lambda st: rng.rand01(st), lambda st: (st, jnp.zeros((), jnp.float32)), state
            )
            state, theme = jax.lax.cond(
                spawn,
                lambda st: rng.randn(st, BARRIER_THEMES),
                lambda st: (st, jnp.zeros((), jnp.int32)),
                state,
            )
            min_y = 2 * agent.ry + BARRIER_RADIUS + 0.5
            barrier_y = y_draw * (BOTTOM_MARGIN - min_y - BARRIER_RADIUS) + min_y
            barrier_x = x_draw * (WORLD_DIM - 2 * BARRIER_RADIUS) + BARRIER_RADIUS
            barrier = engine.entity(
                barrier_x,
                barrier_y,
                rx=BARRIER_RADIUS,
                ry=BARRIER_RADIUS / BARRIER_ASPECTS[jnp.clip(theme, 0, BARRIER_THEMES - 1)],
                type=BARRIER,
                image_type=BARRIER,
                image_theme=theme,
                health=BARRIER_HEALTH,
                collides_with_entities=True,
                active=spawn,
            )
            pool = write_slot(pool, slot, barrier, spawn)
            slot = slot + spawn

        return BossfightState(
            grid=jnp.full((WORLD_DIM * WORLD_DIM,), SPACE, jnp.int32),
            agent=agent,
            entities=pool,
            shields_are_up=jnp.asarray(True),
            health=round_health * num_rounds,
            round_num=jnp.zeros((), jnp.int32),
            round_health=round_health,
            num_rounds=num_rounds,
            attack_modes=attack_modes,
            attack_mode=attack_modes[0],
            invulnerable_duration=invulnerable,
            time_to_swap=invulnerable,
            curr_vel_timeout=jnp.asarray(BOSS_VEL_TIMEOUT, jnp.int32),
            damaged_until_time=jnp.zeros((), jnp.int32),
            last_fire_time=jnp.zeros((), jnp.int32),
            player_laser_theme=player_laser_theme,
            boss_laser_theme=boss_laser_theme,
            boss_slot=jnp.zeros((), jnp.int32),
            shield_slot=jnp.asarray(1, jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            key=key,
            rng=state,
        )

    # -- game rules --------------------------------------------------------- #
    def step_env_state(self, env_state: BossfightState, action: jax.Array):
        key, subkey = jax.random.split(env_state.key)
        step_rand_int = jax.random.randint(subkey, (), 0, 1_000_000)
        now = env_state.steps + 1
        grid = env_state.grid
        agent = env_state.agent
        pool = env_state.entities
        state = env_state.rng

        move_action = jnp.where(action >= 9, 4, action % 9)
        special_action = jnp.where(action >= 9, action - 8, 0)
        action_vx = move_action // 3 - 1
        action_vy = move_action % 3 - 1

        # ---- the player ------------------------------------------------------- #
        vx = (1 - MIXRATE) * agent.vx + MIXRATE * MAX_SPEED * action_vx
        vy = (1 - MIXRATE) * agent.vy + MIXRATE * MAX_SPEED * action_vy
        agent = agent._replace(
            vx=0.9 * vx,
            vy=0.9 * vy,
            is_reflected=jnp.where(action_vx > 0, False, jnp.where(action_vx < 0, True, agent.is_reflected)),
        )
        agent, _ = engine.step_object(
            agent,
            grid.reshape(WORLD_DIM, WORLD_DIM),
            out_of_bounds=jnp.asarray(OUT_OF_BOUNDS),
            is_blocked=lambda tile, is_horizontal: tile == OUT_OF_BOUNDS,
            will_reflect=lambda tile: jnp.asarray(False),
            action_vx=action_vx,
            action_vy=action_vy,
            step_rand_int=step_rand_int,
            max_sub_steps=MAX_SUB_STEPS,
        )

        # every body drifts before the boss picks its next velocity
        pool = jax.vmap(engine.entity_step)(pool)

        # ---- the shield rides on the boss -------------------------------------- #
        pool = pool._replace(
            x=pool.x.at[env_state.shield_slot].set(pool.x[env_state.boss_slot]),
            y=pool.y.at[env_state.shield_slot].set(pool.y[env_state.boss_slot]),
        )
        pool = pool._replace(active=pool.active.at[env_state.shield_slot].set(env_state.shields_are_up))

        # ---- the four draws the original takes every step ---------------------- #
        state, rand_pct = rng.rand01(state)
        state, rand_fire_pct = rng.rand01(state)
        state, rand_pct_x = rng.rand01(state)
        state, rand_pct_y = rng.rand01(state)

        boss_x = pool.x[env_state.boss_slot]
        boss_y = pool.y[env_state.boss_slot]

        # ---- the boss drifts to a new spot and the shield swaps ---------------- #
        relocate = env_state.curr_vel_timeout <= 0
        dest_x = rand_pct_x * (WORLD_DIM - 2 * BOSS_RADIUS) + BOSS_RADIUS
        dest_y = rand_pct_y * (WORLD_DIM - 2 * BOSS_RADIUS - BOTTOM_MARGIN) + BOSS_RADIUS + BOTTOM_MARGIN
        pool = pool._replace(
            vx=pool.vx.at[env_state.boss_slot].set(
                jnp.where(relocate, (dest_x - boss_x) / BOSS_VEL_TIMEOUT, pool.vx[env_state.boss_slot])
            ),
            vy=pool.vy.at[env_state.boss_slot].set(
                jnp.where(relocate, (dest_y - boss_y) / BOSS_VEL_TIMEOUT, pool.vy[env_state.boss_slot])
            ),
        )
        curr_vel_timeout = jnp.where(relocate, jnp.asarray(BOSS_VEL_TIMEOUT, jnp.int32), env_state.curr_vel_timeout - 1)

        swap_ready = relocate & (env_state.time_to_swap <= 0)
        shields_are_up = jnp.where(swap_ready, ~env_state.shields_are_up, env_state.shields_are_up)
        time_to_swap = jnp.where(
            swap_ready,
            jnp.where(
                env_state.shields_are_up,
                jnp.asarray(VULNERABLE_DURATION, jnp.int32),
                env_state.invulnerable_duration,
            ),
            jnp.where(
                relocate & (env_state.time_to_swap > 0),
                env_state.time_to_swap - 1,
                env_state.time_to_swap,
            ),
        )
        attack_mode = jnp.where(
            swap_ready,
            env_state.attack_modes[jnp.clip(env_state.round_num, 0, MAX_ROUNDS - 1)],
            env_state.attack_mode,
        )

        # ---- the player's laser ----------------------------------------------- #
        firing = (special_action == 1) & ((now - env_state.last_fire_time) >= FIRE_COOLDOWN)
        laser = engine.entity(
            agent.x,
            agent.y,
            0.0,
            PLAYER_BULLET_VEL,
            rx=BULLET_RADIUS,
            ry=BULLET_RADIUS,
            type=PLAYER_BULLET,
            image_type=PLAYER_BULLET,
            image_theme=env_state.player_laser_theme,
            collides_with_entities=True,
            expire_time=PLAYER_BULLET_EXPIRE,
            active=True,
        )
        free = ~pool.active
        slot = jnp.argmax(free)
        pool = jax.lax.cond(
            firing & free.any(),
            lambda p: jax.tree.map(lambda buf, value: buf.at[slot].set(value), p, laser),
            lambda p: p,
            pool,
        )
        last_fire_time = jnp.where(firing, now, env_state.last_fire_time)

        # ---- the boss's attack -------------------------------------------------- #
        damaged = env_state.damaged_until_time >= now
        attack_angles, attack_active = attack_shots(attack_mode, now, rand_pct)
        passive = ~shields_are_up & ~damaged & (rand_fire_pct < BASE_FIRE_PROB)
        live_shots = jnp.where(
            damaged,
            jnp.zeros((MAX_SHOTS,), jnp.bool_),
            jnp.where(
                shields_are_up,
                attack_active,
                jnp.arange(MAX_SHOTS) == 0,
            ),
        )
        live_shots = live_shots & jnp.where(shields_are_up, True, passive)
        shot_angles = jnp.where(shields_are_up, attack_angles, jnp.pi * (1 + rand_pct))

        def place(shot, carry):
            pool, used = carry
            live = live_shots[shot] & (used < MAX_ENTITIES)
            bullet = engine.entity(
                boss_x,
                boss_y,
                BOSS_BULLET_VEL * jnp.cos(shot_angles[shot]),
                BOSS_BULLET_VEL * jnp.sin(shot_angles[shot]),
                rx=BOSS_BULLET_RADIUS,
                ry=BOSS_BULLET_RADIUS,
                type=ENEMY_BULLET,
                image_type=ENEMY_BULLET,
                image_theme=env_state.boss_laser_theme,
                expire_time=BOSS_BULLET_EXPIRE,
                active=True,
            )
            free = ~pool.active
            target = jnp.argmax(free)
            write = live & free.any()
            pool = jax.lax.cond(
                write,
                lambda p: jax.tree.map(lambda buf, value: buf.at[target].set(value), p, bullet),
                lambda p: p,
                pool,
            )
            return pool, used + live

        pool, _ = jax.lax.fori_loop(0, MAX_SHOTS, place, (pool, jnp.asarray(0, jnp.int32)))

        # ---- lasers against the shield, the boss and the meteors ---------------- #
        is_laser = pool.active & (pool.type == PLAYER_BULLET)
        reflect_theta = jnp.pi * (1.25 + 0.5 * rand_pct)

        def laser_step(index, carry):
            (
                pool,
                health,
                round_num,
                damaged_until,
                shields_now,
                mode_now,
                time_swap,
                reward,
                finished,
            ) = carry
            shot = jax.tree.map(lambda field: field[index], pool)
            live = is_laser[index]
            hit = engine.collides(shot, pool) & pool.active & live
            hit = hit.at[index].set(False)
            shield_hit = (hit & (pool.type == SHIELDS)).any()
            boss_hit = (hit & (pool.type == BOSS)).any()
            other_hit = (hit & ((pool.type == BARRIER) | (pool.type == ENEMY_BULLET))).any()

            # a shielded laser bounces off with a fresh random heading
            bounced = shield_hit & shields_now
            pool = pool._replace(
                type=pool.type.at[index].set(jnp.where(bounced, REFLECTED_BULLET, pool.type[index])),
                vx=pool.vx.at[index].set(
                    jnp.where(bounced, PLAYER_BULLET_VEL * jnp.cos(reflect_theta) * 0.5, pool.vx[index])
                ),
                vy=pool.vy.at[index].set(
                    jnp.where(bounced, PLAYER_BULLET_VEL * jnp.sin(reflect_theta) * 0.5, pool.vy[index])
                ),
                active=pool.active.at[index].set(pool.active[index] & ~((boss_hit & ~shields_now) | other_hit)),
            )

            scored = boss_hit & ~shields_now
            new_health = health - scored.astype(jnp.int32)
            round_done = scored & (new_health % env_state.round_health == 0)
            round_finished = round_done & (new_health == 0)
            advanced = round_done & ~round_finished
            round_num = round_num + advanced.astype(jnp.int32)
            damaged_until = jnp.where(advanced, now + BOSS_DAMAGED_TIMEOUT, damaged_until)
            shields_now = jnp.where(advanced, True, shields_now)
            mode_now = jnp.where(
                advanced,
                env_state.attack_modes[jnp.clip(round_num, 0, MAX_ROUNDS - 1)],
                mode_now,
            )
            time_swap = jnp.where(advanced, env_state.invulnerable_duration, time_swap)
            reward = reward + POSITIVE_REWARD * round_done.astype(jnp.float32)
            return (
                pool,
                new_health,
                round_num,
                damaged_until,
                shields_now,
                mode_now,
                time_swap,
                reward,
                finished | round_finished,
            )

        carry = (
            pool,
            env_state.health,
            env_state.round_num,
            env_state.damaged_until_time,
            shields_are_up,
            attack_mode,
            time_to_swap,
            jnp.zeros((), jnp.float32),
            jnp.asarray(False),
        )
        carry = jax.lax.fori_loop(0, MAX_ENTITIES, laser_step, carry)
        (
            pool,
            health,
            round_num,
            damaged_until_time,
            shields_are_up,
            attack_mode,
            time_to_swap,
            reward,
            finished,
        ) = carry
        pool = pool._replace(active=pool.active.at[env_state.shield_slot].set(shields_are_up))

        # ---- contact with the player -------------------------------------------- #
        touching = engine.collides(agent, pool) & pool.active
        deadly = touching & ((pool.type == BOSS) | (pool.type == BARRIER) | (pool.type == ENEMY_BULLET))
        terminated = deadly.any() | finished
        reward = reward + jnp.where(finished, COMPLETION_BONUS, 0.0)

        stepped = BossfightState(
            grid=grid,
            agent=agent,
            entities=pool,
            shields_are_up=shields_are_up,
            health=health,
            round_num=round_num,
            round_health=env_state.round_health,
            num_rounds=env_state.num_rounds,
            attack_modes=env_state.attack_modes,
            attack_mode=attack_mode,
            invulnerable_duration=env_state.invulnerable_duration,
            time_to_swap=time_to_swap,
            curr_vel_timeout=curr_vel_timeout,
            damaged_until_time=damaged_until_time,
            last_fire_time=last_fire_time,
            player_laser_theme=env_state.player_laser_theme,
            boss_laser_theme=env_state.boss_laser_theme,
            boss_slot=env_state.boss_slot,
            shield_slot=env_state.shield_slot,
            steps=env_state.steps + 1,
            key=key,
            rng=state,
        )
        return stepped, reward.astype(jnp.float32), terminated

    # -- State construction ------------------------------------------------- #
    def state_from(
        self,
        env_state: BossfightState,
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

    def observe(self, env_state: BossfightState) -> jax.Array:
        """``center_agent`` is off, so the whole arena is on screen."""
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
            WORLD_DIM,
            jnp.asarray(OUT_OF_BOUNDS),
            background=SPACE_COLOUR,
        )
