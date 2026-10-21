"""Tile-based observation rendering.

ProcGen renders sprite assets with Qt; this module draws flat coloured tiles
instead. The framing (an agent-centred window scaled to 64x64 RGB) is the same
shape and layout, so the encoder in the training code is unchanged.
"""

import jax
import jax.numpy as jnp

from procgen_jax.objects import WALL

TILE_PX = 8
VIEW_TILES = 8
OBS_SIZE = TILE_PX * VIEW_TILES  # 64

AGENT_COLOR = jnp.array([220.0, 70.0, 70.0], dtype=jnp.float32)

# Indexed by tile id; PLAYER is never written into the grid.
TILE_COLORS = jnp.array(
    [
        [24.0, 24.0, 40.0],  # SPACE
        [140.0, 140.0, 150.0],  # WALL
        [245.0, 205.0, 60.0],  # GOAL
        [220.0, 70.0, 70.0],  # PLAYER
    ],
    dtype=jnp.float32,
)


def render_observation(grid: jax.Array, agent: jax.Array, world_dim: int) -> jax.Array:
    """Render an agent-centred ``VIEW_TILES`` window, upscaled to 64x64x3 uint8."""
    half = VIEW_TILES // 2
    x0 = agent[0] - half
    y0 = agent[1] - half

    ys = y0 + jnp.arange(VIEW_TILES)
    xs = x0 + jnp.arange(VIEW_TILES)

    inside_y = (ys >= 0) & (ys < world_dim)
    inside_x = (xs >= 0) & (xs < world_dim)
    inside = inside_y[:, None] & inside_x[None, :]

    # Clamp before gathering so the (masked-out) gather stays in bounds.
    tiles = grid[jnp.clip(ys, 0, world_dim - 1)][:, jnp.clip(xs, 0, world_dim - 1)]
    tiles = jnp.where(inside, tiles, WALL)

    pixels = TILE_COLORS[tiles]
    image = jnp.repeat(jnp.repeat(pixels, TILE_PX, axis=0), TILE_PX, axis=1)

    # The agent is centred in its own window, so its tile is always visible.
    ax = jnp.clip(agent[0] - x0, 0, VIEW_TILES - 1) * TILE_PX
    ay = jnp.clip(agent[1] - y0, 0, VIEW_TILES - 1) * TILE_PX
    rows = jnp.arange(OBS_SIZE)[:, None]
    cols = jnp.arange(OBS_SIZE)[None, :]
    agent_mask = (rows >= ay) & (rows < ay + TILE_PX) & (cols >= ax) & (cols < ax + TILE_PX)
    image = jnp.where(agent_mask[..., None], AGENT_COLOR, image)

    return image.astype(jnp.uint8)


# --------------------------------------------------------------------------- #
# scene rendering (float positions, sprite-free)
# --------------------------------------------------------------------------- #
def _colour_table(entries: dict[int, tuple[int, int, int]], default=(0, 0, 0)) -> jax.Array:
    """Dense ``(64, 3)`` uint8 table indexed by object id."""
    table = jnp.tile(jnp.asarray(default, jnp.uint8), (64, 1))
    for index, rgb in entries.items():
        table = table.at[index].set(jnp.asarray(rgb, jnp.uint8))
    return table


#: Coinrun tile ids; the name avoids clashing with the maze table above.; ``ENEMY_BARRIER`` is invisible, like the original.
COINRUN_TILE_COLORS = _colour_table(
    {
        1: (255, 205, 60),  # GOAL: the coin
        15: (120, 70, 40),  # WALL_MID
        16: (90, 160, 60),  # WALL_TOP
        17: (160, 30, 20),  # LAVA_MID
        18: (240, 110, 30),  # LAVA_TOP
        20: (170, 120, 60),  # CRATE (drawn as an entity, kept for completeness)
    },
    default=(110, 145, 255),  # SPACE: sky
)

#: Entity ids; entity 0 is the player, which shares its id with ``SPACE``.
COINRUN_ENTITY_COLORS = _colour_table(
    {
        0: (220, 70, 70),  # PLAYER
        2: (200, 200, 210),  # SAW
        5: (150, 70, 200),  # ENEMY
        20: (170, 120, 60),  # CRATE
    }
)


def render_scene(
    grid: jax.Array,
    center: jax.Array,
    entities,
    view_tiles: int,
    out_of_bounds,
    obs_size: int = OBS_SIZE,
) -> jax.Array:
    """Render a ``view_tiles``-wide window centred on ``center``.

    Unlike :func:`render_observation` this keeps sub-tile positions: each pixel is
    mapped back to world coordinates, so bodies of width ``2 * rx`` land where the
    physics says they are.  Entity layer ``render_z`` is painted above the grid in
    ascending order, matching ``draw_foreground``.
    """
    unit = obs_size / view_tiles
    cols = jnp.arange(obs_size)[None, :]
    rows = jnp.arange(obs_size)[:, None]

    world_x = center[0] - view_tiles / 2 + (cols + 0.5) / unit
    world_y = center[1] + view_tiles / 2 - (rows + 0.5) / unit

    tiles = _tile_lookup(grid, world_x, world_y, out_of_bounds)
    image = COINRUN_TILE_COLORS[tiles]

    half_x = jnp.abs(world_x - entities.x[:, None, None]) < entities.rx[:, None, None]
    half_y = jnp.abs(world_y - entities.y[:, None, None]) < entities.ry[:, None, None]
    covered = half_x & half_y & entities.active[:, None, None]

    for layer in (0, 1):
        painted = covered & (entities.render_z[:, None, None] == layer)
        hit = painted.any(axis=0)
        index = jnp.argmax(painted, axis=0)
        colour = COINRUN_ENTITY_COLORS[entities.type[index]]
        image = jnp.where(hit[..., None], colour, image)

    return image.astype(jnp.uint8)


def _tile_lookup(grid: jax.Array, world_x: jax.Array, world_y: jax.Array, out_of_bounds) -> jax.Array:
    """Tile ids under a 2-D grid of world coordinates, padded with ``out_of_bounds``."""
    height, width = grid.shape
    xi = jnp.floor(world_x).astype(jnp.int32)
    yi = jnp.floor(world_y).astype(jnp.int32)
    inside = (world_x >= 0) & (world_y >= 0) & (xi < width) & (yi < height)
    flat = grid.reshape(-1)[jnp.clip(yi, 0, height - 1) * width + jnp.clip(xi, 0, width - 1)]
    return jnp.where(inside, flat, out_of_bounds)
