"""Record one episode of each game.

    pixi run video           # -> runs/*.gif   (ignored by git)
    pixi run video docs      # -> docs/*.gif   (the copies in the README)

Frames are produced by a single jitted ``lax.scan`` and handed to ``ffmpeg`` as
raw RGB, so nothing beyond JAX/NumPy is needed.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from procgen_jax import CoinRunEnv, MazeEnv
from procgen_jax.recording import encode_video, rollout

STEPS = 200
FPS = 20


def random_actions(key, steps, action_size):
    return jax.random.randint(key, (steps,), 0, action_size)


def run_and_run(key, steps, action_size):
    """Walk right, and jump periodically -- enough to get moving in coinrun."""
    actions = jnp.full((steps,), 7, jnp.int32)
    actions = actions.at[::5].set(5)  # slot 5 is the jump button
    return actions


def record(name, env, actions, out_dir: Path) -> Path:
    state = env.reset(jax.random.PRNGKey(0))
    frames, rewards, dones = jax.jit(lambda acts: rollout(env, state, acts))(actions)

    path = out_dir / f'{name}.gif'
    encode_video(path, frames, fps=FPS, scale=4)
    print(
        f'{name}: {frames.shape[0]} frames, {int(dones.sum())} episodes ended, '
        f'return {float(rewards.sum()):.1f} -> {path} ({path.stat().st_size // 1024} KiB)'
    )
    return path


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')
    out_dir.mkdir(exist_ok=True)

    maze = MazeEnv(world_dim=15)
    record('maze', maze, random_actions(jax.random.PRNGKey(1), STEPS, maze.action_size), out_dir)

    coinrun = CoinRunEnv()
    record('coinrun', coinrun, run_and_run(jax.random.PRNGKey(1), STEPS, coinrun.action_size), out_dir)


if __name__ == '__main__':
    main()
