"""The package has to be installable: data shipped, version single-sourced, all games.

These are the checks that would have caught the wheel shipping without the sprite
atlas, which is the one way this repo can look fine in a checkout and broken once
installed.
"""

import pathlib
import tomllib

import pytest

import procgen_jax
from procgen_jax import games

ROOT = pathlib.Path(__file__).resolve().parent.parent


def project_metadata():
    with open(ROOT / 'pyproject.toml', 'rb') as handle:
        return tomllib.load(handle)['project']


def test_version_matches_pyproject():
    assert procgen_jax.__version__ == project_metadata()['version']


def test_sprite_atlas_is_importable_data():
    from importlib.resources import files

    import numpy as np

    atlas = files('procgen_jax') / 'data' / 'sprites.npz'
    assert atlas.is_file(), 'sprites.npz must ship inside the package'

    with np.load(atlas) as archive:
        assert archive['pixels'].ndim == 4
        assert archive['pixels'].shape[1:] == (32, 32, 4)
        assert len(archive['names']) > 200


def test_every_game_is_exported():
    expected = {
        'MazeEnv',
        'CoinRunEnv',
        'BigFishEnv',
        'LeaperEnv',
        'PlunderEnv',
        'HeistEnv',
        'StarPilotEnv',
        'DodgeballEnv',
        'FruitBotEnv',
        'JumperEnv',
        'ChaserEnv',
        'CaveFlyerEnv',
        'NinjaEnv',
        'ClimberEnv',
        'MinerEnv',
        'BossfightEnv',
    }
    exported = {name for name in dir(games) if name.endswith('Env')}

    assert expected <= exported


def test_license_and_readme_are_present():
    metadata = project_metadata()

    assert (ROOT / 'LICENSE').is_file()
    assert (ROOT / metadata['readme']).is_file()


@pytest.mark.parametrize('name', ['MazeEnv', 'CoinRunEnv', 'BossfightEnv'])
def test_the_entry_points_build(name):
    env = getattr(procgen_jax, name)()

    assert env.action_size > 0
    assert len(env.observation_shape) == 3
