"""Engine primitives, transcribed from ``basic-abstract-game.cpp``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import engine

WALL_TOP = 16
WALL_MID = 15


def flat_grid(width=16, height=16, floor=WALL_TOP):
    return jnp.zeros((height, width), jnp.int32).at[0, :].set(floor)


def blocked(tile, is_horizontal):
    return (tile == WALL_TOP) | (tile == WALL_MID)


def never(tile):
    return jnp.asarray(False)


def move(ent, grid, action_vx=0, action_vy=0, step_rand_int=3):
    return engine.step_object(
        ent,
        grid,
        out_of_bounds=jnp.asarray(WALL_MID),
        is_blocked=blocked,
        will_reflect=never,
        action_vx=jnp.asarray(action_vx),
        action_vy=jnp.asarray(action_vy),
        step_rand_int=jnp.asarray(step_rand_int),
        max_sub_steps=6,
    )


def test_tile_at_reads_the_grid_and_pads_outside():
    grid = flat_grid()

    assert int(engine.tile_at(grid, 0.5, 0.5, 99)) == WALL_TOP
    assert int(engine.tile_at(grid, 1.5, 5.5, 99)) == 0
    # Outside the world -- including negative coordinates -- is out of bounds.
    assert int(engine.tile_at(grid, -0.5, 0.5, 99)) == 99
    assert int(engine.tile_at(grid, 0.5, -0.5, 99)) == 99
    assert int(engine.tile_at(grid, 16.5, 0.5, 99)) == 99


def test_sub_step_count_is_bounded_by_the_velocity():
    slow = engine.entity(1.0, 1.0, vx=0.0, vy=0.0)
    fast = engine.entity(1.0, 1.0, vx=0.5, vy=1.5)

    assert int(engine.num_sub_steps_for(slow, 6)) == 4  # floored at 4
    assert int(engine.num_sub_steps_for(fast, 6)) == 6  # int(4 * 1.581) == 6


def test_a_fast_body_does_not_tunnel_through_the_floor():
    """Sub-stepping is what stops ``vy = -1.5`` from skipping a tile."""
    grid = flat_grid()
    ent = engine.entity(1.5, 1.5787, vy=-1.5, rx=0.5, ry=0.5787)

    moved, _ = move(ent, grid, action_vy=-1)

    assert np.isclose(float(moved.y), 1.5787, atol=1e-5)
    assert float(moved.vy) == 0.0


def test_body_rests_exactly_on_the_tile_boundary():
    grid = flat_grid()
    ent = engine.entity(1.5, 1.5787, vy=-0.2, rx=0.5, ry=0.5787)

    moved, _ = move(ent, grid)

    # The floor's top edge is y = 1.0, so the centre rests at 1.0 + ry.
    assert np.isclose(float(moved.y), 1.0 + 0.5787, atol=1e-5)


def test_walls_block_horizontal_movement():
    grid = flat_grid().at[1, 2].set(WALL_MID)
    ent = engine.entity(1.5, 1.5787, vx=0.5, rx=0.5, ry=0.5787)

    moved, _ = move(ent, grid, action_vx=1)

    assert float(moved.x) < 2.0
    assert float(moved.vx) == 0.0


def test_reflection_flips_velocity_and_mirrors_the_body():
    grid = flat_grid().at[1, 2].set(WALL_MID)
    enemy = engine.entity(1.5, 1.5, vx=0.3, rx=0.5, ry=0.5)

    moved, _ = engine.step_object(
        enemy,
        grid,
        out_of_bounds=jnp.asarray(WALL_MID),
        is_blocked=blocked,
        will_reflect=lambda tile: tile == WALL_MID,
        action_vx=jnp.asarray(0),
        action_vy=jnp.asarray(0),
        step_rand_int=jnp.asarray(0),
        max_sub_steps=4,
    )

    assert float(moved.vx) < 0  # bounced
    assert float(moved.x) <= 1.5  # pushed back off the wall


@pytest.mark.parametrize('depth', [0.0, 4.0, 8.0])
def test_out_of_bounds_is_only_true_once_fully_outside(depth):
    width = height = 8
    inside = engine.entity(4.0, 4.0, rx=0.5, ry=0.5)
    left = engine.entity(-0.6, 4.0, rx=0.5, ry=0.5)

    assert not bool(engine.is_out_of_bounds(inside, width, height))
    assert bool(engine.is_out_of_bounds(left, width, height))


def test_collision_uses_the_aabb_convention():
    a = engine.entity(1.0, 1.0, rx=0.5, ry=0.5)
    b = engine.entity(1.9, 1.0, rx=0.5, ry=0.5)
    c = engine.entity(2.1, 1.0, rx=0.5, ry=0.5)

    assert bool(engine.collides(a, b))
    assert not bool(engine.collides(a, c))


def test_position_stays_finite_under_a_long_random_walk():
    grid = flat_grid()
    ent = engine.entity(8.5, 8.5, rx=0.5, ry=0.5)

    def body(carry, action):
        moved, _ = move(carry, grid, action_vx=action // 3 - 1, action_vy=action % 3 - 1)
        return moved, moved.y

    ent, ys = jax.jit(lambda e: jax.lax.scan(body, e, jnp.arange(16) % 9))(ent)

    assert jnp.all(jnp.isfinite(ys))
    assert jnp.all(ys >= 0.0)
