"""ProcGen games, one module each."""

from procgen_jax.games.bigfish import BigFishEnv
from procgen_jax.games.coinrun import CoinRunEnv
from procgen_jax.games.heist import HeistEnv
from procgen_jax.games.leaper import LeaperEnv
from procgen_jax.games.maze import MazeEnv
from procgen_jax.games.plunder import PlunderEnv

__all__ = ['BigFishEnv', 'CoinRunEnv', 'HeistEnv', 'LeaperEnv', 'MazeEnv', 'PlunderEnv']
