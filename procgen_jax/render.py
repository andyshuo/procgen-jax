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
