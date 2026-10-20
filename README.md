# procgen-jax

[OpenAI ProcGen](https://github.com/openai/procgen) reimplemented in **pure JAX**.

ProcGen ships as a C++ extension driven from Python: every `step()` crosses the
Python/C++ boundary, and no part of the environment can be `jit`ed. Here the level
generator, the game rules and the observation renderer are all `jax.numpy` / `lax`
code, so an entire vectorised rollout becomes a single `lax.scan` — no host↔device
synchronisation per step, and `jax.vmap` turnaround for free.

## Status

| Game       | Rules ported | Notes                                    |
| ---------- | ------------ | ---------------------------------------- |
| `maze`     | ✅            | Kruskal mazes, grid movement, goal reward |

The remaining 15 games are planned; see the roadmap below. Each game in the
original is only 121-523 lines of logic on top of a shared engine, so the
incremental cost per game is small once the engine primitives settle.

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
the last frame of the finished episode.

`MazeState` holds `grid` (the tile map), `agent` (x, y), `steps` and `key`, so an
episode is a pure function of its starting key.

## Implementation notes

- **Level generation.** Kruskal's algorithm on a union-find array, following
  `procgen/src/mazegen.cpp`. Cells sit on even coordinates, walls between them on
  odd ones. Everything runs inside `lax.scan` / `lax.while_loop`, so levels are
  differentiable-free but still jit-able.
- **Actions.** ProcGen encodes the 3x3 movement grid as `vx = a // 3 - 1`,
  `vy = a % 3 - 1`; `maze` additionally zeroes `vy` whenever `vx != 0`.
- **Rewards.** +10 for reaching the goal (ProcGen: `REWARD = 10.0`), timeout 500.
- **Observations.** An agent-centred 8x8 tile window upscaled to 64x64 RGB, padded
  with walls outside the world.

### Performance

`pixi run demo` drives a batch of 64 environments for 256 steps inside one compiled
`lax.scan`. On a CPU-only machine this reaches ~44k environment steps/s; there is no
per-step host synchronisation because the whole rollout is a single XLA program.

## Known deviations from the original

These are deliberate first-milestone choices, not bugs:

1. **Rendering uses flat tiles** instead of Kenney sprite assets. Observation shape,
   framing and dtype match, so downstream encoders are unaffected.
2. **The maze always fills the world.** ProcGen randomises `maze_dim` per episode
   and centres it inside `world_dim`; here `maze_dim == world_dim`.
3. **Actions 9-14 are unused.** ProcGen exposes 15 discrete actions but only 0-8 are
   meaningful on the 3x3 grid.
4. Levels are *not* bit-identical to the C++ implementation, because the RNG
   sequence differs. The generation algorithm is the same.

## Roadmap

- [ ] Port the remaining 15 games, starting with `coinrun` (platformer physics).
- [ ] Shared engine primitives (`Entity`, `is_blocked`, grid/entity collisions).
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
