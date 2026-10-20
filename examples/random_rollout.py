"""Random-policy rollout: checks shape/dtype end-to-end and reports throughput."""

import time

import jax
import jax.numpy as jnp

from procgen_jax import MazeEnv
from procgen_jax.maze import vmap_reset


def main() -> None:
    env = MazeEnv(world_dim=15)
    batch, steps = 64, 256

    @jax.jit
    def rollout(key):
        keys = jax.random.split(key, batch)
        state = vmap_reset(env, keys)
        actions = jax.random.randint(key, (steps, batch), 0, 9)

        def body(carry, action):
            state, total = carry
            state, reward, done = jax.vmap(env.step)(state, action)
            return (state, total + reward), done

        return jax.lax.scan(body, (state, jnp.zeros((batch,))), actions)

    key = jax.random.PRNGKey(0)

    # First call compiles; time the second one.
    (_, totals), dones = rollout(key)
    jax.block_until_ready(totals)

    start = time.perf_counter()
    repetitions = 10
    for i in range(repetitions):
        (_, totals), dones = rollout(jax.random.fold_in(key, i))
    jax.block_until_ready(totals)
    elapsed = time.perf_counter() - start

    total_steps = repetitions * batch * steps
    observation = env.observation(env.reset(key))

    print(f'observation shape {observation.shape} dtype {observation.dtype}')
    print(f'batch {batch} x {steps} steps x {repetitions} rollouts')
    print(f'{total_steps / elapsed:,.0f} env steps/s')
    print(f'episode returns: {totals.mean():.2f} mean, {totals.max():.2f} max')
    print(f'done fraction per step: {dones.mean():.4f}')


if __name__ == '__main__':
    main()
