# procgen-jax

[OpenAI ProcGen](https://github.com/openai/procgen) reimplemented in **pure JAX**.

ProcGen ships as a C++ extension driven from Python: every `step()` crosses the
Python/C++ boundary, and no part of the environment can be `jit`ed. Here the level
generator, the game rules and the observation renderer are all `jax.numpy` / `lax`
code, so an entire vectorised rollout becomes a single `lax.scan` — no host↔device
synchronisation per step, and `jax.vmap` turnaround for free.

## Status

| Game       | Rules ported | Notes                                                       |
| ---------- | ------------ | ----------------------------------------------------------- |
| `maze`     | ✅            | Kruskal mazes, grid movement, goal reward                    |
| `coinrun`  | ✅            | platformer physics, lava/saw/enemy deaths, coin, crates      |
|            |              | levels are verified completable by a greedy policy           |

The remaining 14 games are planned; see the roadmap below. Each game in the
original is only 121-523 lines of logic on top of a shared engine, so the
incremental cost per game is small now that `procgen_jax/engine.py` exists.

## Install

```bash
pixi install          # CPU JAX
pixi install -e gpu   # CUDA 12 JAX
```

## Usage

```python
import jax
from procgen_jax import MazeEnv

env = MazeEnv(world_dim=15)  # 15 = ProcGen's "easy" maze
state = env.reset(jax.random.PRNGKey(0))

for _ in range(500):
    action = jax.random.randint(jax.random.PRNGKey(1), (), 0, 9)
    state = env.step(state, action)  # state.obs is (64, 64, 3) uint8
    print(float(state.reward), bool(state.done), state.info['terminated'])
```

Vectorised, and fully jit-able:

```python
import jax
from procgen_jax import MazeEnv

env = MazeEnv(world_dim=15)


@jax.jit
def rollout(keys):
    state = jax.vmap(env.reset)(keys)  # (B, ...) pytree
    actions = jax.random.randint(keys[0], (128, keys.shape[0]), 0, 9)

    def body(carry, action):
        state = jax.vmap(env.step)(carry, action)
        return state, state.reward

    states, rewards = jax.lax.scan(body, state, actions)
    return states.obs, rewards


observations, rewards = rollout(jax.random.split(jax.random.PRNGKey(0), 64))
```

Run the bundled demo (random policy, prints throughput):

```bash
pixi run demo
```

Record a video:

```bash
pixi run video        # writes runs/maze.gif and runs/coinrun.gif
```

`procgen_jax/recording.py` collects the frames with one `lax.scan` (the observation
lives in the state, so an episode is a `(T, 64, 64, 3)` array with no per-step host
round-trip) and pipes them to `ffmpeg` as raw RGB -- no Python imaging dependency.
Recording is deliberately *not* a `Wrapper`: a wrapper that accumulated frames would
either grow a pytree leaf every step, which breaks `scan`'s static carry, or call
back into the host, which breaks `jit`.

## Environment API

The interface follows Brax's `Env`/`State` design rather than a gym-style
`(obs, reward, done, info)` tuple: the observation lives *inside* the state, so a
rollout is a pure `lax.scan` over pytrees with no per-step host round-trip.

| Method                    | Returns                          |
| ------------------------- | -------------------------------- |
| `env.reset(key)`          | `State`                          |
| `env.step(state, action)` | `State`                          |

`State` is a `NamedTuple` with `env_state` (the game-specific state), `obs`
(`uint8[64, 64, 3]`), `reward`, `done` and `info`.

Auto-reset is built in: on the step where an episode ends, `env_state` and `obs`
are the *fresh episode*, while `done` (and `info['terminated']` /
`info['truncated']`) still describe the transition that just finished. That is
gymnax's semantics, and it keeps `scan` shapes static. `info['final_obs']` keeps
the last frame the env emitted for that episode — the one the agent acted on, so a
terminal event stays visible — which auto-reset would otherwise overwrite.

`MazeState` holds `grid` (the tile map), `agent` (x, y), `steps` and `key`, so an
episode is a pure function of its starting key.

## Implementation notes

- **Level generation.** Kruskal's algorithm on a union-find array, following
  `procgen/src/mazegen.cpp`. Cells sit on even coordinates, walls between them on
  odd ones. Everything runs inside `lax.scan` / `lax.while_loop`, so levels are
  differentiable-free but still jit-able.
- **Actions.** ProcGen encodes the 3x3 movement grid as `vx = a // 3 - 1`,
  `vy = a % 3 - 1`; `maze` additionally zeroes `vy` whenever `vx != 0`.
- **Auto-reset.** `step` resets through `procgen_jax.base.auto_reset`, a `lax.cond`
  around the level generator, so a running episode never pays for regeneration.
  `done` still describes the transition, gymnax-style.
- **Rewards.** +10 for reaching the goal (ProcGen: `REWARD = 10.0`), timeout 500.
- **Observations.** An agent-centred tile window upscaled to 64x64 RGB, padded with
  walls outside the world. `maze` shows 8 tiles; `coinrun` shows the 13 tiles its
  constructor sets via `visibility`.

### Engine (`procgen_jax/engine.py`)

Transcribed from `procgen/src/basic-abstract-game.cpp`, and shared by every game:

- **Entities** are fixed-size pytrees with an `active` mask instead of a
  `std::vector`, so a level with a variable number of enemies still has a static
  shape.
- **Sub-stepping.** `step_object` splits a displacement into `int(4 * |v|)`
  pieces (floored at 4) so a fast body cannot tunnel through a tile. The C++ trip
  count is data dependent, so the loop runs the maximum number of times and masks
  the rest -- same arithmetic, static shape.
- **Collision.** `sub_step` samples the four corners of the moved body against the
  grid, then either reflects it (enemies bouncing off walls) or snaps it flush
  against the tile it hit. Both are branch-free `where` expressions.
- **Gravity is applied before movement**, and jumping is an impulse clamped to
  `max_jump`. This is why the jump peak is the *discrete* sum
  `1.5 + 1.3 + ... + 0.1 = 6.4` tiles rather than the continuous `v^2 / 2g = 5.6`.

Not ported: entity pushing (the `push_obj` recursion, which only matters when
movable bodies block each other) and trails.

### Recording

`procgen_jax/recording.py` has `rollout(env, state, actions)`, which returns
`(frames, rewards, dones)` from a single `lax.scan`, and `encode_video(path, frames)`,
which pipes `uint8` frames to `ffmpeg` as raw RGB. `pixi run video` writes
`runs/maze.gif` and `runs/coinrun.gif`.

### Performance

`pixi run demo` drives a batch of 64 environments for 256 steps inside one compiled
`lax.scan`; on this CPU box that reaches ~20k environment steps/s. A single
un-batched jitted `scan` runs at ~47k steps/s: `lax.cond` lets it skip the level
generator while an episode is running, whereas `vmap` lowers the same call to a
`select` and evaluates both branches. Nothing crosses to the host per step — the
whole rollout is one XLA program.

## Known deviations from the original

These are deliberate first-milestone choices, not bugs:

1. **Rendering uses flat tiles** instead of Kenney sprite assets. Observation shape,
   framing and dtype match, so downstream encoders are unaffected.
2. **The maze always fills the world.** ProcGen randomises `maze_dim` per episode
   and centres it inside `world_dim`; here `maze_dim == world_dim`.
3. **Actions 9-14 are no-ops.** ProcGen exposes 15 discrete actions for every game
   but the 3x3 grid games only decode 0-8, so the extra slots are inert here instead
   of being snapped onto the nearest move.
4. Levels are *not* bit-identical to the C++ implementation, because the RNG
   sequence differs. The generation algorithm is the same.
5. **`coinrun` observations are always agent-centred**, whereas ProcGen makes it an
   option; entity pushing is not ported, and `EasyMode` (`distribution_mode`) is not
   implemented, so every level uses the harder distribution.

## Roadmap

- [x] `coinrun` -- platformer physics, so the engine has a second consumer.
- [x] Shared engine primitives (`Entity`, `is_blocked`, sub-stepped collisions).
- [ ] Port the remaining 14 games (next: `jumper`/`climber`, same physics family).
- [ ] Sprite-based rendering close to the original.
- [ ] Level-parity harness against the C++ implementation.
- [ ] PPO baseline + throughput comparison against the original.

## Development

```bash
pixi run test   # pytest
pixi run lint   # ruff check + format check
```

## License

No license has been chosen yet, so no rights to use, modify or redistribute are
granted. Ask the author first.
