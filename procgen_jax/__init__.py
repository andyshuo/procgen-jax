"""Procedurally generated environments, reimplemented in pure JAX.

The interface follows Brax's ``Env`` / ``State`` design: the state carries the
observation, so a whole vectorised rollout compiles into one ``lax.scan``.
"""

from procgen_jax.base import Env, State, Wrapper, auto_reset
from procgen_jax.games.bigfish import BigFishEnv
from procgen_jax.games.caveflyer import CaveFlyerEnv
from procgen_jax.games.chaser import ChaserEnv
from procgen_jax.games.coinrun import CoinRunEnv
from procgen_jax.games.dodgeball import DodgeballEnv
from procgen_jax.games.fruitbot import FruitBotEnv
from procgen_jax.games.heist import HeistEnv
from procgen_jax.games.jumper import JumperEnv
from procgen_jax.games.leaper import LeaperEnv
from procgen_jax.games.maze import MazeEnv
from procgen_jax.games.ninja import NinjaEnv
from procgen_jax.games.plunder import PlunderEnv
from procgen_jax.games.starpilot import StarPilotEnv
from procgen_jax.version import __version__

__all__ = [
    'BigFishEnv',
    'CaveFlyerEnv',
    'ChaserEnv',
    'CoinRunEnv',
    'DodgeballEnv',
    'Env',
    'FruitBotEnv',
    'HeistEnv',
    'JumperEnv',
    'LeaperEnv',
    'MazeEnv',
    'NinjaEnv',
    'PlunderEnv',
    'StarPilotEnv',
    'State',
    'Wrapper',
    'auto_reset',
    '__version__',
]
