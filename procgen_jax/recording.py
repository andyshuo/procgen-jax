"""Record a rollout to a video file, with no Python imaging dependency.

The observation lives inside :class:`~procgen_jax.base.State`, so an episode's
frames are just the ``obs`` leaves of one ``lax.scan``: a single device-side
array, one transfer to the host, no per-step callback.

Writing the file is deliberately *not* a ``Wrapper``.  A wrapper that accumulated
frames would either have to grow a pytree leaf every step -- which breaks
``scan``'s static carry -- or call back into the host, which breaks ``jit``.
Recording is a rollout-level concern, so it lives here and takes the env as an
argument.

Frames are piped to ``ffmpeg`` as raw RGB, which keeps the package's dependency
list at JAX + NumPy.
"""

import shutil
import subprocess

import jax
import jax.numpy as jnp


def rollout(env, state, actions):
    """Step ``env`` through ``actions`` and collect ``(frames, rewards, dones)``."""

    def body(carry, action):
        carry = env.step(carry, action)
        return carry, (carry.obs, carry.reward, carry.done)

    _, (frames, rewards, dones) = jax.lax.scan(body, state, actions)
    return frames, rewards, dones


def encode_video(path, frames, fps: int = 20, scale: int = 4) -> str:
    """Write ``(T, H, W, 3)`` uint8 ``frames`` to ``path`` via ffmpeg.

    ``scale`` upscales with nearest-neighbour so flat tiles stay crisp instead of
    turning into gradients.  ``.gif`` gets a generated palette; anything else is
    left to ffmpeg's default encoder for the container.
    """
    frames = jnp.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'expected (T, H, W, 3) frames, got {frames.shape}')
    if frames.size == 0:
        raise ValueError('no frames to encode')

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg was not found on PATH')

    count, height, width, _ = frames.shape
    filters = [f'scale=iw*{scale}:ih*{scale}:flags=neighbor']
    if str(path).endswith('.gif'):
        filters.append('split[a][b];[a]palettegen[p];[b][p]paletteuse')
    command = [
        ffmpeg,
        '-y',
        '-loglevel',
        'error',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgb24',
        '-s',
        f'{width}x{height}',
        '-r',
        str(fps),
        '-i',
        '-',
    ]
    if str(path).endswith('.gif'):
        command += ['-loop', '0']
    command += ['-vf', ','.join(filters), str(path)]

    subprocess.run(command, input=jnp.asarray(frames, jnp.uint8).tobytes(), check=True)
    return str(path)
