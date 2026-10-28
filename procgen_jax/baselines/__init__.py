"""Training baselines for the environments.

``ppo`` is a dependency-free PPO: plain ``jax.numpy`` arrays, a hand-written Adam and
a single ``lax.scan`` around everything, so it can be compiled as one program or run
eagerly to measure what compilation buys.
"""

from procgen_jax.baselines.ppo import (
    PPOConfig,
    compute_gae,
    forward,
    greedy_action,
    init_params,
    make_train,
    normalise_advantages,
    ppo_loss,
    sample_action,
    train,
)

__all__ = [
    'PPOConfig',
    'compute_gae',
    'forward',
    'greedy_action',
    'init_params',
    'make_train',
    'normalise_advantages',
    'ppo_loss',
    'sample_action',
    'train',
]
