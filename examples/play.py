"""Play a game yourself, in the terminal.

    pixi run play                      # coinrun
    pixi run play -- --game maze       # the other one
    pixi run play -- --seed 42         # a level seed (coinrun only)
    pixi run play -- --eager           # no jax.jit at all: one key press = one frame
    pixi run play -- --record runs/me.gif

The observation is drawn with 24-bit colour half blocks (``▄``), so each character
cell shows two pixels: 64x64 becomes 64 columns by 32 rows.

Controls:  a/d or arrows move, w/up/space jump, s stand still, r restart,
           n/p next/previous seed, . single step (also with --eager), q quit.

``--eager`` really does skip ``jax.jit``; it is slow (a step dispatches thousands
of primitives, so expect a couple of seconds per frame) which is why the default
compiles the step once.  Rendering is always eager.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import jax
import jax.numpy as jnp
import numpy as np

from procgen_jax import (
    BigFishEnv,
    CoinRunEnv,
    DodgeballEnv,
    FruitBotEnv,
    HeistEnv,
    JumperEnv,
    LeaperEnv,
    MazeEnv,
    PlunderEnv,
    StarPilotEnv,
)
from procgen_jax.recording import encode_video

CLEAR = '\x1b[H\x1b[2J'
HIDE_CURSOR = '\x1b[?25l'
SHOW_CURSOR = '\x1b[?25h'
RESET = '\x1b[0m'


#: ProcGen encodes the 3x3 grid as ``(vx + 1) * 3 + (vy + 1)``; slot 5 is jump.
def action_of(horizontal: int, jump: bool) -> int:
    return (horizontal + 1) * 3 + (2 if jump else 1)


def half_blocks(frame: np.ndarray) -> str:
    """Two pixels per character cell, using the lower-half block as the glyph."""
    height, width, _ = frame.shape
    if height % 2:
        frame = np.concatenate([frame, np.zeros((1, width, 3), np.uint8)], axis=0)
    top = frame[0::2].astype(np.int32)
    bottom = frame[1::2].astype(np.int32)
    lines = []
    for y in range(top.shape[0]):
        line = []
        for x in range(width):
            t = top[y, x]
            b = bottom[y, x]
            line.append(f'\x1b[38;2;{b[0]};{b[1]};{b[2]}m\x1b[48;2;{t[0]};{t[1]};{t[2]}m▄')
        line.append(RESET)
        lines.append(''.join(line))
    return '\n'.join(lines)


def ascii_art(frame: np.ndarray, ramp=' .:-=+*#%@') -> str:
    """Fallback for terminals without colour: brightness ramp, 2x2 pixels per cell."""
    grey = frame.astype(np.float32).mean(axis=-1)
    height, width = grey.shape
    rows = []
    for y in range(0, height - 1, 2):
        cols = []
        for x in range(0, width - 1, 2):
            value = grey[y : y + 2, x : x + 2].mean() / 255.0
            cols.append(ramp[min(len(ramp) - 1, int(value * len(ramp)))])
        rows.append(''.join(cols))
    return '\n'.join(rows)


class Keyboard:
    """Single key presses in cbreak mode, without curses."""

    def __init__(self, stream):
        self.stream = stream
        self.fd = stream.fileno()
        self.saved = None

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def read(self, timeout: float) -> list[str]:
        """Key presses since the last call, newest last.

        A single ``read`` can return several bytes -- a fast typist, a terminal
        repeating a held key, or an escape sequence arriving in one chunk -- so the
        buffer is split into individual keys instead of being matched whole.
        """
        # Always poll the descriptor: a frame can take longer than the frame
        # interval (especially with --eager), and a negative timeout must mean
        # "do not wait", not "do not look".
        if not select.select([self.stream], [], [], max(0.0, timeout))[0]:
            return []
        data = os.read(self.fd, 64).decode('utf-8', 'ignore')

        arrows = {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}
        keys, index = [], 0
        while index < len(data):
            if data[index] == '\x1b' and data[index + 1 : index + 2] == '[':
                keys.append(arrows.get(data[index + 2 : index + 3], ''))
                index += 3
            else:
                keys.append(data[index])
                index += 1
        return [key for key in keys if key]


def build(game: str, seed):
    if game == 'maze':
        env = MazeEnv(world_dim=15)
    elif game == 'bigfish':
        env = BigFishEnv(level_seed=seed)
    elif game == 'leaper':
        env = LeaperEnv(level_seed=seed)
    elif game == 'plunder':
        env = PlunderEnv(level_seed=seed)
    elif game == 'heist':
        env = HeistEnv(level_seed=seed)
    elif game == 'starpilot':
        env = StarPilotEnv(level_seed=seed)
    elif game == 'dodgeball':
        env = DodgeballEnv(level_seed=seed)
    elif game == 'fruitbot':
        env = FruitBotEnv(level_seed=seed)
    elif game == 'jumper':
        env = JumperEnv(level_seed=seed)
    else:
        env = CoinRunEnv(level_seed=seed)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--game',
        default='coinrun',
        choices=[
            'coinrun',
            'maze',
            'bigfish',
            'leaper',
            'plunder',
            'heist',
            'starpilot',
            'dodgeball',
            'fruitbot',
            'jumper',
        ],
    )
    parser.add_argument('--seed', type=int, default=None, help='pin the level seed')
    parser.add_argument('--eager', action='store_true', help='do not jit the step (slow but literal)')
    parser.add_argument('--ascii', action='store_true', help='no colour, brightness ramp instead')
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--record', default=None, help='write the frames to this .gif/.mp4')
    parser.add_argument('--auto', type=int, default=0, help='play N steps headlessly, then exit')
    args = parser.parse_args()

    env = build(args.game, args.seed)
    step = env.step if args.eager else jax.jit(env.step)
    reset = env.reset if args.eager else jax.jit(env.reset)
    key_state = jax.random.PRNGKey(0)
    state = reset(key_state)

    render = ascii_art if args.ascii else half_blocks
    interval = 1.0 / args.fps
    horizontal, jump = 0, False
    paused, running = False, True
    frames, seed = [], args.seed

    def restart(new_seed):
        nonlocal env, step, reset, state, key_state
        env = build(args.game, new_seed)
        step = env.step if args.eager else jax.jit(env.step)
        reset = env.reset if args.eager else jax.jit(env.reset)
        state = reset(key_state)

    if args.auto:
        # Headless smoke test: right, jumping every fifth step (coinrun) or a
        # random policy (maze).  No TTY needed.
        rng = np.random.default_rng(0)
        total = 0.0
        for tick in range(args.auto):
            action = action_of(1, tick % 5 == 0)
            if args.game in (
                'maze',
                'bigfish',
                'leaper',
                'plunder',
                'heist',
                'starpilot',
                'dodgeball',
                'fruitbot',
                'jumper',
            ):
                action = int(rng.integers(0, 9))
            state = step(state, jnp.asarray(action))
            total += float(state.reward)
            frames.append(np.asarray(state.obs))
        if args.record:
            encode_video(args.record, np.stack(frames))
        print(f'{args.auto} steps, return {total:.1f}, seed {seed}')
        return 0

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    keyboard = Keyboard(sys.stdin) if interactive else None
    if keyboard is None:
        print('stdin/stdout are not a TTY; use --auto for a headless run', file=sys.stderr)
        return 2

    def apply(key: str) -> None:
        """One key press, applied to the latched input and the loop state."""
        nonlocal horizontal, jump, paused, running, seed, state
        if key in ('a', 'left'):
            horizontal = -1
        elif key in ('d', 'right'):
            horizontal = 1
        elif key == 's':
            horizontal = 0
        elif key in ('w', 'up', ' '):
            jump = True
        elif key == 'q':
            running = False
        elif key == 'r':
            restart(seed)
            frames.clear()
        elif key in ('n', 'm') and args.game == 'coinrun':
            seed = (seed or 0) + (1 if key == 'n' else -1)
            restart(seed)
            frames.clear()
        elif key == 'p':
            paused = not paused
        elif key == '.':  # one frame at a time, paused or not
            state = step(state, jnp.asarray(action_of(horizontal, jump)))
            jump = False
            frames.append(np.asarray(state.obs))

    with keyboard:
        sys.stdout.write(HIDE_CURSOR)
        next_tick = time.perf_counter()
        try:
            while running:
                for key in keyboard.read(max(0.0, next_tick - time.perf_counter())):
                    apply(key)
                if not running:
                    break

                if not paused:
                    state = step(state, jnp.asarray(action_of(horizontal, jump)))
                    jump = False
                    frames.append(np.asarray(state.obs))

                agent = state.env_state.agent
                if hasattr(agent, 'x'):
                    where = f'({float(agent.x):.2f},{float(agent.y):.2f})'
                else:
                    where = f'({float(agent[0]):.2f},{float(agent[1]):.2f})'

                header = (
                    f'{args.game}  seed={seed}  steps={int(state.env_state.steps)}  '
                    f'return={float(state.reward):.1f}  pos={where}'
                )
                if bool(state.done):
                    header += '  [EPISODE ENDED]'
                if paused:
                    header += '  [paused: . to step, p to resume]'
                sys.stdout.write(CLEAR + header + '\n' + render(np.asarray(state.obs)) + '\n')
                sys.stdout.flush()

                next_tick += interval
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
        finally:
            sys.stdout.write(SHOW_CURSOR + RESET)
            sys.stdout.flush()

    if args.record and frames:
        encode_video(args.record, np.stack(frames))
        print(f'recorded {len(frames)} frames to {args.record}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
