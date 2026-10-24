"""Procedurally generated environments, reimplemented in pure JAX.

The interface follows Brax's ``Env`` / ``State`` design: the state carries the
observation, so a whole vectorised rollout compiles into one ``lax.scan``.
"""

from procgen_jax.base import Env, State, Wrapper, auto_reset
from procgen_jax.games.bigfish import BigFishEnv
from procgen_jax.games.coinrun import CoinRunEnv
from procgen_jax.games.heist import HeistEnv
from procgen_jax.games.leaper import LeaperEnv
from procgen_jax.games.maze import MazeEnv
from procgen_jax.games.plunder import PlunderEnv
from procgen_jax.version import __version__

__all__ = [
    'BigFishEnv',
    'CoinRunEnv',
    'Env',
    'HeistEnv',
    'LeaperEnv',
    'MazeEnv',
    'PlunderEnv',
    'State',
    'Wrapper',
    'auto_reset',
    '__version__',
]
