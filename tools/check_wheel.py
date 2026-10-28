"""Check that an *installed* copy of the package is usable.

A checkout can look perfectly healthy while the wheel is broken: if the sprite atlas
is not declared as package data, ``import procgen_jax`` still works and only the first
``reset`` fails, which is exactly the kind of thing a release should never ship.

Run it with the package you want to test on ``sys.path``::

    python tools/check_wheel.py                     # the checkout
    pip install dist/*.whl && python tools/check_wheel.py
"""

import sys

import numpy as np

import procgen_jax
from procgen_jax import sprites


def main() -> int:
    package_dir = procgen_jax.__path__[0]
    print(f'procgen_jax {procgen_jax.__version__} at {package_dir}')

    atlas = np.load(f'{package_dir}/data/sprites.npz')
    pixels = atlas['pixels']
    if pixels.shape[1:] != (32, 32, 4):
        raise SystemExit(f'sprite atlas has the wrong shape: {pixels.shape}')
    if len(atlas['names']) != pixels.shape[0]:
        raise SystemExit('sprite atlas names and pixels disagree')
    print(f'sprite atlas: {pixels.shape[0]} sprites, {pixels.shape[1]}x{pixels.shape[2]}')

    import jax

    for name, kwargs in (('MazeEnv', {'world_dim': 15}), ('CoinRunEnv', {}), ('BossfightEnv', {})):
        env = getattr(procgen_jax, name)(**kwargs)
        state = env.reset(jax.random.PRNGKey(0))
        if state.obs.shape != (64, 64, 3):
            raise SystemExit(f'{name} produced {state.obs.shape}, expected (64, 64, 3)')
        print(f'{name}: obs {state.obs.shape} {state.obs.dtype}, sprites from {sprites.__file__}')

    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
