"""ProcGen games, one module each."""

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

__all__ = [
    'BigFishEnv',
    'CaveFlyerEnv',
    'ChaserEnv',
    'CoinRunEnv',
    'DodgeballEnv',
    'FruitBotEnv',
    'HeistEnv',
    'JumperEnv',
    'LeaperEnv',
    'MazeEnv',
    'NinjaEnv',
    'PlunderEnv',
    'StarPilotEnv',
]
