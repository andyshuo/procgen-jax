"""Environment interface, modelled on Brax's ``Env`` / ``State`` design.

The important convention is that :class:`State` carries the observation: ``step``
returns a whole new state, so a compiled rollout is just

    state = jax.lax.scan(lambda s, a: (env.step(s, a), s.reward), state, actions)[0]

with no separate "render the observation" call and no host round-trip per step.

Like Brax's ``State`` this is a pytree; unlike Brax we use ``typing.NamedTuple``
instead of ``flax.struct.dataclass`` so the package only depends on JAX/NumPy.
"""

import abc
from collections.abc import Sequence
from typing import Any, NamedTuple

import jax


class State(NamedTuple):
    """A single environment transition.

    Attributes
    ----------
    env_state:
        Game-specific state (grid, agent, counters, RNG key, ...).
    obs:
        Observation produced by this transition (post auto-reset, so on a ``done``
        step this is the first frame of the next episode).
    reward:
        Scalar reward for the transition.
    done:
        Whether the episode ended (``terminated or truncated``). It describes the
        transition, following gymnax, so it stays ``True`` even though
        ``env_state``/``obs`` have already been reset.
    info:
        Auxiliary data: ``terminated``, ``truncated`` and ``final_obs`` (the last
        observation the env emitted for the finished episode, which auto-reset
        would otherwise overwrite).
    """

    env_state: Any
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class Env(abc.ABC):
    """Interface for a functional JAX environment."""

    @abc.abstractmethod
    def reset(self, key: jax.Array) -> State:
        """Start a new episode from ``key``."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep. The returned state is auto-reset when ``done``."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """Number of discrete actions expected by :meth:`step`."""

    @property
    @abc.abstractmethod
    def observation_shape(self) -> Sequence[int]:
        """Shape of the observation array."""

    @property
    def observation_size(self) -> int:
        """Flattened observation size, matching Brax's ``observation_size``."""
        size = 1
        for dim in self.observation_shape:
            size *= int(dim)
        return size

    @property
    def unwrapped(self) -> 'Env':
        return self


class Wrapper(Env):
    """Base class for environment transformations (reward shaping, stats, ...)."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, key: jax.Array) -> State:
        return self.env.reset(key)

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self.env.observation_shape

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        # Only reached for attributes this wrapper does not define itself.
        if name == 'env' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.env, name)
