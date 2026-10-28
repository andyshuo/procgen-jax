"""Procedurally generated environments, reimplemented in pure JAX.

The interface follows Brax's ``Env`` / ``State`` design: the state carries the
observation, so a whole vectorised rollout compiles into one ``lax.scan``.
"""

from procgen_jax.base import Env, State, Wrapper, auto_reset
from procgen_jax.games.bigfish import BigFishEnv
from procgen_jax.games.bossfight import BossfightEnv
from procgen_jax.games.caveflyer import CaveFlyerEnv
from procgen_jax.games.chaser import ChaserEnv
from procgen_jax.games.climber import ClimberEnv
from procgen_jax.games.coinrun import CoinRunEnv
from procgen_jax.games.dodgeball import DodgeballEnv
from procgen_jax.games.fruitbot import FruitBotEnv
from procgen_jax.games.heist import HeistEnv
from procgen_jax.games.jumper import JumperEnv
from procgen_jax.games.leaper import LeaperEnv
from procgen_jax.games.maze import MazeEnv
from procgen_jax.games.miner import MinerEnv
from procgen_jax.games.ninja import NinjaEnv
from procgen_jax.games.plunder import PlunderEnv
from procgen_jax.games.starpilot import StarPilotEnv
from procgen_jax.version import __version__

#: Game name -> env class, keyed the way the original ProcGen CLI spells them.
ENVIRONMENTS = {
    'bigfish': BigFishEnv,
    'bossfight': BossfightEnv,
    'caveflyer': CaveFlyerEnv,
    'chaser': ChaserEnv,
    'climber': ClimberEnv,
    'coinrun': CoinRunEnv,
    'dodgeball': DodgeballEnv,
    'fruitbot': FruitBotEnv,
    'heist': HeistEnv,
    'jumper': JumperEnv,
    'leaper': LeaperEnv,
    'maze': MazeEnv,
    'miner': MinerEnv,
    'ninja': NinjaEnv,
    'plunder': PlunderEnv,
    'starpilot': StarPilotEnv,
}


def make_env(name: str, **kwargs) -> Env:
    """Build a game by name, e.g. ``make_env('maze', world_dim=15)``."""
    try:
        return ENVIRONMENTS[name](**kwargs)
    except KeyError:
        raise ValueError(f'unknown game {name!r}; choose from {sorted(ENVIRONMENTS)}') from None


__all__ = [
    'BigFishEnv',
    'CaveFlyerEnv',
    'BossfightEnv',
    'ChaserEnv',
    'ClimberEnv',
    'CoinRunEnv',
    'DodgeballEnv',
    'ENVIRONMENTS',
    'Env',
    'FruitBotEnv',
    'HeistEnv',
    'JumperEnv',
    'LeaperEnv',
    'MazeEnv',
    'MinerEnv',
    'NinjaEnv',
    'PlunderEnv',
    'StarPilotEnv',
    'State',
    'Wrapper',
    'auto_reset',
    'make_env',
    '__version__',
]
