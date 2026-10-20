"""Procedurally generated, pure-JAX maze environments.

A from-scratch JAX implementation of OpenAI ProcGen's `maze` game. Everything —
level generation, game rules and observation rendering — is expressed with
`jax.numpy` and `lax` primitives, so a whole vectorised rollout can be compiled
into a single `lax.scan`.
"""

from procgen_jax.maze import MazeEnv, MazeState
from procgen_jax.version import __version__

__all__ = ['MazeEnv', 'MazeState', '__version__']
