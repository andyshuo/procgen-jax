"""Grid/entity physics, transcribed from ProcGen's ``procgen/src/basic-abstract-game.cpp``.

The C++ engine is an in-place, object-oriented simulator: entities live in a
``std::vector``, the grid is a mutable ``TileMap``, and collision resolution walks
``for`` loops whose trip counts depend on float velocities.  A faithful JAX port
therefore keeps the *arithmetic* (same constants, same order of operations, same
snapping rules) and changes the *representation*:

* entities are fixed-size pytrees with an ``active`` mask instead of a vector;
* every loop has a static trip count plus a mask (:func:`step_object` replays
  ``max_sub_steps`` iterations and masks out the ones the engine would skip);
* positions and velocities stay ``float32`` and follow the same order of
  operations, so a trajectory matches the original up to float rounding.

Not ported: entity pushing (``sub_step``'s ``push_obj`` recursion, which only
matters when movable entities block each other), sprite/asset handling and
serialisation.  ``Entity::step``'s friction/decay is a no-op for every game
ported so far (``friction == 1``, ``grow_rate == 1``), so it is omitted.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: ``object-ids.h``: the agent is entity type 0 (``PLAYER``).
PLAYER = 0
#: ``object-ids.h``: the id that out-of-grid lookups resolve to before a game
#: overrides it.  Games normally set ``out_of_bounds_object`` to a wall.
WALL_OBJ = 51

#: A small negative bias so a body exactly on a tile boundary counts as inside.
POS_EPS = -0.001
#: Corner sampling factor used by ``sub_step`` when testing the grid.
COLLISION_MARGIN = 0.98


class Entity(NamedTuple):
    """One body. Batched entities carry a leading dimension on every field."""

    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    rx: jax.Array
    ry: jax.Array
    type: jax.Array
    render_z: jax.Array
    smart_step: jax.Array
    active: jax.Array


def entity(
    x,
    y,
    vx=0.0,
    vy=0.0,
    rx=0.5,
    ry=None,
    type=PLAYER,
    render_z=0,
    smart_step=False,
    active=True,
) -> Entity:
    """Scalar :class:`Entity` constructor, mirroring ``add_entity``/``add_entity_rxy``."""
    return Entity(
        x=jnp.asarray(x, dtype=jnp.float32),
        y=jnp.asarray(y, dtype=jnp.float32),
        vx=jnp.asarray(vx, dtype=jnp.float32),
        vy=jnp.asarray(vy, dtype=jnp.float32),
        rx=jnp.asarray(rx, dtype=jnp.float32),
        ry=jnp.asarray(rx if ry is None else ry, dtype=jnp.float32),
        type=jnp.asarray(type, dtype=jnp.int32),
        render_z=jnp.asarray(render_z, dtype=jnp.int32),
        smart_step=jnp.asarray(smart_step),
        active=jnp.asarray(active),
    )


def tile_at(grid: jax.Array, x: jax.Array, y: jax.Array, out_of_bounds: jax.Array) -> jax.Array:
    """``get_obj_from_floats``: the tile under ``(x, y)``, or ``out_of_bounds``."""
    height, width = grid.shape
    xi = jnp.floor(x).astype(jnp.int32)
    yi = jnp.floor(y).astype(jnp.int32)
    inside = (x >= 0) & (y >= 0) & (xi < width) & (yi < height)
    flat = grid.reshape(-1)[jnp.clip(yi, 0, height - 1) * width + jnp.clip(xi, 0, width - 1)]
    return jnp.where(inside, flat, out_of_bounds)


def is_out_of_bounds(ent: Entity, width, height) -> jax.Array:
    """``is_out_of_bounds``: a body has completely left the world."""
    return (ent.x + ent.rx < 0) | (ent.y + ent.ry < 0) | (ent.x - ent.rx > width) | (ent.y - ent.ry > height)


def clamp_abs(velocity: jax.Array, limit) -> jax.Array:
    """``clip_abs``: clamp the magnitude of a scalar velocity."""
    return jnp.clip(velocity, -limit, limit)


def num_sub_steps_for(ent: Entity, maximum: int) -> jax.Array:
    """``num_sub_steps = int(4 * |v|)``, floored at 4 and capped at ``maximum``.

    The C++ trip count depends on the velocity, which a jitted loop cannot use
    directly; we compute the same count and let :func:`step_object` mask the
    iterations beyond it.
    """
    magnitude = jnp.sqrt(ent.vx * ent.vx + ent.vy * ent.vy)
    count = jnp.maximum(jnp.asarray(4, jnp.int32), jnp.floor(4 * magnitude).astype(jnp.int32))
    return jnp.minimum(count, jnp.asarray(maximum, jnp.int32))


def _snap(new: jax.Array, r: jax.Array, delta: jax.Array) -> jax.Array:
    """Push a body flush against the tile boundary it just crossed."""
    return jnp.where(delta > 0, jnp.floor(new + r) - r, jnp.ceil(new - r) + r)


def sub_step(
    ent: Entity,
    grid: jax.Array,
    dx: jax.Array,
    dy: jax.Array,
    *,
    out_of_bounds: jax.Array,
    is_blocked,
    will_reflect,
    blocked_by_entities=None,
):
    """One collision-resolved displacement, mirroring ``BasicAbstractGame::sub_step``.

    ``is_blocked(tile, is_horizontal)`` and ``will_reflect(tile)`` receive a tile
    type.  ``blocked_by_entities(ent, is_horizontal)`` runs after the grid has
    been resolved and returns ``(blocked, is_on_support)``.  Returns the moved
    entity, whether the move was blocked and whether the body landed on a support.
    """
    is_horizontal = dx != 0
    nx = ent.x + dx
    ny = ent.y + dy

    def corner(i, j):
        return tile_at(
            grid,
            nx + ent.rx * COLLISION_MARGIN * (2 * i - 1),
            ny + ent.ry * COLLISION_MARGIN * (2 * j - 1),
            out_of_bounds,
        )

    corners = jnp.stack([corner(i, j) for i in (0, 1) for j in (0, 1)])
    block = jnp.stack([is_blocked(t, is_horizontal) for t in corners]).any()
    reflect = jnp.stack([will_reflect(t) for t in corners]).any()

    # Reflect: mirror the body across the tile boundary it hit.
    delta_x = jnp.where(
        dx < 0,
        jnp.ceil(nx - ent.rx) - (nx - ent.rx),
        jnp.floor(nx + ent.rx) - (nx + ent.rx),
    )
    delta_y = jnp.where(
        dy < 0,
        jnp.ceil(ny - ent.ry) - (ny - ent.ry),
        jnp.floor(ny + ent.ry) - (ny + ent.ry),
    )
    nx_reflected = jnp.where(is_horizontal, nx + 2 * delta_x, nx)
    ny_reflected = jnp.where(is_horizontal, ny, ny + 2 * delta_y)

    # Block: snap the body flush against the boundary (``grid_step`` is off).
    nx_blocked = jnp.where(is_horizontal, _snap(nx, ent.rx, dx), nx)
    ny_blocked = jnp.where(is_horizontal, ny, _snap(ny, ent.ry, dy))

    # ``if (reflect) ... else if (block)``: reflection wins over blocking.
    nx = jnp.where(reflect, nx_reflected, jnp.where(block, nx_blocked, nx))
    ny = jnp.where(reflect, ny_reflected, jnp.where(block, ny_blocked, ny))
    vx = jnp.where(reflect & is_horizontal, -ent.vx, ent.vx)
    vy = jnp.where(reflect & ~is_horizontal, -ent.vy, ent.vy)

    moved = ent._replace(x=nx, y=ny, vx=vx, vy=vy)

    blocked2 = jnp.asarray(False)
    support = jnp.asarray(False)
    if blocked_by_entities is not None:
        blocked2, support = blocked_by_entities(moved, is_horizontal)

    return moved, (block | blocked2), support


def _choose_axis(ent: Entity, action_vx, action_vy, step_rand_int) -> jax.Array:
    """Horizontal-first ordering, with the agent's action overriding a tie.

    Ties are broken randomly (``step_rand_int % 2``) so paths through tight
    corridors are unbiased; the agent's own action decides its axis instead.
    """
    cmp = jnp.abs(ent.vx) - jnp.abs(ent.vy)
    step_x_first = jnp.where(cmp == 0, step_rand_int % 2 == 0, cmp > 0)
    step_x_first = jnp.where(action_vx != 0, True, step_x_first)
    step_x_first = jnp.where(action_vy != 0, False, step_x_first)
    return step_x_first


def step_object(
    ent: Entity,
    grid: jax.Array,
    *,
    out_of_bounds: jax.Array,
    is_blocked,
    will_reflect,
    blocked_by_entities=None,
    action_vx: jax.Array,
    action_vy: jax.Array,
    step_rand_int: jax.Array,
    max_sub_steps: int,
):
    """``BasicAbstractGame::basic_step_object``: move one body with sub-stepping.

    Returns ``(moved, landed_on_support)``.  Damping is applied exactly as the
    engine does: ``vx *= vx_pct`` where ``vx_pct`` counts the sub-steps that were
    *not* blocked and is divided by the engine's own trip count.
    """
    count = num_sub_steps_for(ent, max_sub_steps)
    step_x_first = _choose_axis(ent, action_vx, action_vy, step_rand_int)

    def move(e, is_horizontal: bool):
        dx = jnp.where(is_horizontal, e.vx / count, 0.0)
        dy = jnp.where(is_horizontal, 0.0, e.vy / count)
        return sub_step(
            e,
            grid,
            dx,
            dy,
            out_of_bounds=out_of_bounds,
            is_blocked=is_blocked,
            will_reflect=will_reflect,
            blocked_by_entities=blocked_by_entities,
        )

    def body(carry, substep):
        moved, vx_pct, vy_pct, broken, support = carry
        active = (substep < count) & ~broken

        def run(e):
            first = move(e, step_x_first)
            second = move(first[0], ~step_x_first)
            block_first = jnp.where(step_x_first, first[1], second[1])
            block_second = jnp.where(step_x_first, second[1], first[1])
            return second[0], block_first, block_second, first[2] | second[2]

        def skip(e):
            zero = jnp.asarray(False)
            return e, zero, zero, zero

        moved, block_x, block_y, landed = jax.lax.cond(active, run, skip, moved)

        vx_pct = jnp.where(active & ~block_x, vx_pct + 1, vx_pct)
        vy_pct = jnp.where(active & ~block_y, vy_pct + 1, vy_pct)
        broken = broken | (active & block_x & block_y)
        support = support | (active & landed)
        return (moved, vx_pct, vy_pct, broken, support), None

    zero = jnp.asarray(0, jnp.int32)
    init = (ent, zero, zero, jnp.asarray(False), jnp.asarray(False))
    (moved, vx_pct, vy_pct, _, support), _ = jax.lax.scan(body, init, jnp.arange(max_sub_steps), length=max_sub_steps)

    moved = moved._replace(vx=moved.vx * (vx_pct / count), vy=moved.vy * (vy_pct / count))
    return moved, support


def collides(a: Entity, b: Entity, margin=0.0) -> jax.Array:
    """``has_collision``: AABB overlap in the engine's (rx, ry) convention."""
    threshold_x = (a.rx + b.rx) + margin
    threshold_y = (a.ry + b.ry) + margin
    return (jnp.abs(a.x - b.x) < threshold_x) & (jnp.abs(a.y - b.y) < threshold_y)


def touched_tiles(ent: Entity):
    """``check_grid_collisions`` bounds: ``(min_x, max_x, min_y, max_y)``.

    ``int()`` truncates towards zero in C++, so ``jnp.trunc`` is the right cast.
    """
    min_x = jnp.trunc(ent.x - (ent.rx + POS_EPS)).astype(jnp.int32)
    max_x = jnp.trunc(ent.x + (ent.rx + POS_EPS)).astype(jnp.int32)
    min_y = jnp.trunc(ent.y - (ent.ry + POS_EPS)).astype(jnp.int32)
    max_y = jnp.trunc(ent.y + (ent.ry + POS_EPS)).astype(jnp.int32)
    return min_x, max_x, min_y, max_y
