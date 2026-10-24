"""ProcGen games, one module each."""

from procgen_jax.games.bigfish import BigFishEnv
from procgen_jax.games.coinrun import CoinRunEnv
from procgen_jax.games.maze import MazeEnv

__all__ = ['BigFishEnv', 'CoinRunEnv', 'MazeEnv']
