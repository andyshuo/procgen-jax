"""Sprite atlas: ProcGen's Kenney artwork as a JAX-friendly array.

``tools/build_atlas.py`` downloads the PNGs and pipes them through ``ffmpeg`` into
a single ``(N, SIZE, SIZE, 4)`` uint8 array, so rendering is a plain gather and the
package still only depends on JAX + NumPy.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

#: Atlas file, built by ``pixi run atlas``.
ATLAS_FILE = Path(__file__).parent / 'data' / 'sprites.npz'
#: Every sprite is squashed to this square; the renderer stretches them back.
SIZE = 32


def load(path: Path = ATLAS_FILE):
    """Return ``(pixels, index, aspects)``: atlas, name -> row, width/height."""
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing; run `pixi run atlas` to download the Kenney assets')
    data = np.load(path)
    pixels = jnp.asarray(data['pixels'])
    aspects = jnp.asarray(data['aspects'], jnp.float32)
    index = {str(name): row for row, name in enumerate(data['names'])}
    return pixels, index, aspects


PIXELS, INDEX, ASPECTS = load()


def row(name: str) -> int:
    """Atlas row for a sprite name such as ``Ground/Grass/mid``."""
    return INDEX[name]
