"""Train the PPO baseline on a ProcGen game, and measure what jit buys.

    pixi run train -- --game maze --updates 200
    pixi run train -- --game maze --compare          # jit vs the same program eager
    pixi run train -- --game coinrun --eager --updates 5

``--compare`` runs the *same* ``make_train`` program twice: once wrapped in
``jax.jit`` and once called directly.  Nothing else changes, so the difference is
purely compilation -- one fused ``lax.scan`` against per-primitive dispatch.
"""

import argparse
import time

import jax
import jax.numpy as jnp

from procgen_jax import ENVIRONMENTS, make_env
from procgen_jax.baselines import PPOConfig, greedy_action, make_train


def build(game: str, level_seed: int | None):
    kwargs = {'world_dim': 15} if game == 'maze' else {}
    if level_seed is not None:
        kwargs['level_seed'] = level_seed
    return make_env(game, **kwargs)


def run(env, cfg, jit: bool, repeats: int = 1):
    """Time the training loop ``repeats`` times; returns ``(elapsed, metrics, params)``.

    The runner is built (and therefore compiled) once, outside the timer for every
    repeat after the first.  ``repeats=2`` on the jitted path therefore reports
    steady-state throughput rather than compilation.
    """
    runner = make_train(env, cfg)
    if jit:
        runner = jax.jit(runner)

    timing = 0.0
    metrics = params = None
    for index in range(repeats):
        start = time.perf_counter()
        metrics, params = runner(jax.random.PRNGKey(cfg.seed + index))
        jax.block_until_ready(metrics['loss'])
        timing = time.perf_counter() - start
    return timing, metrics, params


def evaluate(env, params, cfg: PPOConfig, key, steps: int = 1000):
    """Greedy rollout in one environment: ``(total reward, steps)``."""

    def body(carry, key):
        state, total, count = carry
        action = greedy_action(params, state.obs[None], cfg)[0]
        stepped = env.step(state, action)
        return (stepped, total + stepped.reward, count + 1), None

    state = env.reset(key)
    (_, total, count), _ = jax.lax.scan(
        body, (state, jnp.zeros(()), jnp.zeros((), jnp.int32)), jax.random.split(key, steps)
    )
    return total, count


def env_only(env, cfg: PPOConfig, reps: int = 5):
    """Throughput of the environment alone: ``(eager_steps_per_s, jit_steps_per_s)``.

    This is the number the repository is actually about -- the policy update below is
    a few small matmuls and is dispatch-bound in both modes, so it dilutes the ratio.
    """

    def rollout(key):
        state = jax.vmap(env.reset)(jax.random.split(key, cfg.num_envs))

        def body(state, step_key):
            actions = jax.random.randint(step_key, (cfg.num_envs,), 0, env.action_size)
            return jax.vmap(env.step)(state, actions), None

        state, _ = jax.lax.scan(body, state, jax.random.split(key, cfg.rollout_len))
        return state.reward

    compiled = jax.jit(rollout)
    compiled(jax.random.PRNGKey(0)).block_until_ready()  # pay for compilation

    def time_of(run, wait):
        start = time.perf_counter()
        for index in range(reps):
            out = run(jax.random.PRNGKey(index))
            if wait:
                out.block_until_ready()
        return (time.perf_counter() - start) / reps

    steps = cfg.num_envs * cfg.rollout_len
    eager = time_of(rollout, False)
    jitted = time_of(compiled, True)
    return steps / eager, steps / jitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', default='maze', choices=sorted(ENVIRONMENTS))
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--rollout', type=int, default=64)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatches', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--level-seed', type=int, default=None)
    parser.add_argument('--compare', action='store_true', help='time the jitted and eager runs and report the ratio')
    parser.add_argument('--eager', action='store_true', help='run without jit (slow, for reference)')
    parser.add_argument('--env-only', action='store_true', help='only time the environment, no learner')
    args = parser.parse_args()

    cfg = PPOConfig(
        num_envs=args.envs,
        rollout_len=args.rollout,
        total_updates=args.updates,
        epochs=args.epochs,
        num_minibatches=args.minibatches,
        learning_rate=args.lr,
        seed=args.seed,
    )
    steps = cfg.total_updates * cfg.batch_size
    env = build(args.game, args.level_seed)
    evaluate_jit = jax.jit(lambda params, key: evaluate(env, params, cfg, key))

    if args.env_only:
        eager_rate, jit_rate = env_only(env, cfg)
        print(f'eager {eager_rate:9,.0f} env steps/s')
        print(f'  jit {jit_rate:9,.0f} env steps/s')
        print(
            f'\njit is {jit_rate / eager_rate:,.1f}x faster on {args.game} ({cfg.num_envs} envs x {cfg.rollout_len} steps)'
        )
        return

    modes = [False, True] if args.compare else [not args.eager]
    results = {}
    for jit in modes:
        label = 'jit' if jit else 'eager'
        # jit runs twice so the reported number is steady state, not compilation
        elapsed, metrics, _ = run(env, cfg, jit, repeats=2 if jit else 1)
        results[label] = elapsed
        results[f'{label}_params'] = _
        curve = metrics['episode_return']
        print(
            f'{label:>5}: {elapsed:7.2f}s  {steps / elapsed:9,.0f} steps/s  '
            f'loss {float(metrics["loss"][-1]):+.4f}  '
            f'entropy {float(metrics["entropy"][-1]):.3f}  '
            f'reward/step {float(jnp.mean(curve[:10])):+.4f} -> {float(jnp.mean(curve[-10:])):+.4f}'
        )

    if args.compare:
        eager, jitted = results['eager'], results['jit']
        print(f'\njit is {eager / jitted:,.1f}x faster than eager on {args.game} ({steps:,} steps)')

    total, count = evaluate_jit(
        results['jit_params' if 'jit_params' in results else 'eager_params'], jax.random.PRNGKey(999)
    )
    print(f'greedy eval after training: total reward {total:+.1f} over {count} steps ({total / count:+.4f} per step)')


if __name__ == '__main__':
    main()
