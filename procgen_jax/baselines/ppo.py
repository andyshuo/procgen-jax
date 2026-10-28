"""A small PPO baseline in plain ``jax.numpy``.

No Flax, no Optax: the policy is a pytree of arrays, the optimizer is Adam written
out, and the whole training loop -- vectorised rollout, GAE, clipped update -- is one
``lax.scan``.  That is deliberate.  The point of this repository is that an
environment step can live inside a compiled loop, and a baseline that pulls in a
framework would hide whether that actually pays off.

Frames are 64x64 RGB uint8.  Each one is average-pooled 4x4 to 16x16x3 and flattened
to 768 features, then two tanh layers produce the action logits and the value.  This
is the "small" baseline: it is cheap enough to train on a CPU while still solving
``maze`` and learning something on the platformers.

Usage::

    from procgen_jax import MazeEnv
    from procgen_jax.baselines import PPOConfig, train

    cfg = PPOConfig(num_envs=8, rollout_len=128, total_updates=100)
    metrics, params = train(MazeEnv(world_dim=15), cfg, jax.random.PRNGKey(0))
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

Params = dict[str, jax.Array]


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters. Defaults are the ones ``examples/train_ppo.py`` reports."""

    num_envs: int = 8
    rollout_len: int = 64
    total_updates: int = 100
    epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: tuple[int, ...] = (128, 128)
    pooling: int = 4
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.rollout_len * self.num_envs

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    def __post_init__(self) -> None:
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f'rollout_len * num_envs ({self.batch_size}) must divide into {self.num_minibatches} minibatches'
            )


# --------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------- #
def _features(obs: jax.Array, pooling: int) -> jax.Array:
    """``(..., 64, 64, 3)`` uint8 -> ``(..., 768)`` float32, average-pooled."""
    frames = obs.astype(jnp.float32) / 255.0
    leading = frames.shape[:-3]
    size = frames.shape[-3]
    blocks = size // pooling
    frames = frames.reshape(*leading, blocks, pooling, blocks, pooling, frames.shape[-1])
    frames = frames.mean(axis=(-4, -2))
    return frames.reshape(*leading, blocks * blocks * frames.shape[-1])


def init_params(key: jax.Array, obs_shape=(64, 64, 3), num_actions: int = 9, cfg=None) -> Params:
    """He-initialised tanh MLP with small policy head and unit value head."""
    cfg = cfg or PPOConfig()
    features = (obs_shape[0] // cfg.pooling) * (obs_shape[1] // cfg.pooling) * obs_shape[2]
    sizes = [features, *cfg.hidden]
    params: Params = {}
    for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        key, sub = jax.random.split(key)
        params[f'w{index}'] = jax.random.normal(sub, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        params[f'b{index}'] = jnp.zeros((fan_out,))
    key, sub = jax.random.split(key)
    params['policy_w'] = jax.random.normal(sub, (sizes[-1], num_actions)) * 0.01
    params['policy_b'] = jnp.zeros((num_actions,))
    params['value_w'] = jax.random.normal(sub, (sizes[-1], 1)) * 1.0
    params['value_b'] = jnp.zeros((1,))
    return params


def forward(params: Params, obs: jax.Array, cfg: PPOConfig):
    """``(obs, logits, value)``; ``value`` comes back as ``(...,)``."""
    x = _features(obs, cfg.pooling)
    layers = len(cfg.hidden)
    for index in range(layers):
        x = jnp.tanh(x @ params[f'w{index}'] + params[f'b{index}'])
    logits = x @ params['policy_w'] + params['policy_b']
    value = (x @ params['value_w'] + params['value_b'])[..., 0]
    return logits, value


def sample_action(key: jax.Array, logits: jax.Array):
    """Categorical sample plus its log-probability."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    action = jax.random.categorical(key, logits, axis=-1)
    return action, jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def greedy_action(params: Params, obs: jax.Array, cfg: PPOConfig) -> jax.Array:
    logits, _ = forward(params, obs, cfg)
    return jnp.argmax(logits, axis=-1)


# --------------------------------------------------------------------------- #
# advantage estimation
# --------------------------------------------------------------------------- #
def compute_gae(rewards, values, terminated, truncated, final_values, cfg: PPOConfig):
    """Generalised advantages, bootstrapping from ``final_obs`` on truncation.

    ``rewards``/``values``/the masks are ``(T, N)``; ``final_values`` is ``(N,)``, the
    value of the observation each episode ended on.  A *termination* cuts the trace, a
    *truncation* also cuts it but still bootstraps from the final observation (the repo
    keeps it in ``info['final_obs']`` for exactly this).  Returns ``(advantages, returns)``.
    """
    next_values = jnp.concatenate([values[1:], final_values[None]], axis=0)
    alive = 1.0 - (terminated | truncated).astype(jnp.float32)
    not_terminated = 1.0 - terminated.astype(jnp.float32)

    def body(advantage, inputs):
        reward, value, next_value, alive_now, keep_bootstrap = inputs
        delta = reward + cfg.gamma * next_value * keep_bootstrap - value
        advantage = delta + cfg.gamma * cfg.gae_lambda * alive_now * advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        body,
        jnp.zeros_like(values[0]),
        (rewards, values, next_values, alive, not_terminated),
        reverse=True,
    )
    return advantages, advantages + values


# --------------------------------------------------------------------------- #
# the update
# --------------------------------------------------------------------------- #
def clip_by_global_norm(tree: Params, max_norm: float) -> Params:
    """Rescale the whole gradient tree to ``max_norm`` (no-op when it is smaller)."""
    if max_norm <= 0:
        return tree
    leaves = jax.tree.leaves(tree)
    norm = jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda leaf: leaf * scale, tree)


def adam_init(params: Params) -> dict[str, Params]:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {'m': zeros, 'v': zeros, 't': jnp.zeros((), jnp.int32)}


def adam_step(params: Params, grads: Params, state, cfg: PPOConfig):
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = state['t'] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state['m'], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state['v'], grads)
    m_hat = jax.tree.map(lambda m: m / (1 - b1**t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2**t), v)
    params = jax.tree.map(lambda p, mh, vh: p - cfg.learning_rate * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, {'m': m, 'v': v, 't': t}


def ppo_loss(params: Params, batch, cfg: PPOConfig):
    """Clipped surrogate plus value and entropy terms, for one minibatch."""
    logits, value = forward(params, batch['obs'], cfg)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    action_log_probs = jnp.take_along_axis(log_probs, batch['actions'][:, None], axis=-1)[:, 0]
    entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)

    ratio = jnp.exp(action_log_probs - batch['log_probs'])
    unclipped = ratio * batch['advantages']
    clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch['advantages']
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean((value - batch['returns']) ** 2)
    entropy_bonus = jnp.mean(entropy)
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus

    approx_kl = jnp.mean(batch['log_probs'] - action_log_probs)
    clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > cfg.clip_eps)
    return loss, {
        'loss': loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy_bonus,
        'approx_kl': approx_kl,
        'clip_fraction': clip_fraction,
    }


def _update(params, opt_state, batch, key, cfg: PPOConfig):
    """One PPO update: ``epochs`` passes over ``num_minibatches`` shuffled splits."""
    size = batch['obs'].shape[0]
    steps_per_epoch = cfg.num_minibatches

    def epoch_body(carry, epoch_key):
        params, opt_state = carry
        order = jax.random.permutation(epoch_key, size)

        def minibatch_body(carry, index):
            params, opt_state = carry
            start = index * cfg.minibatch_size
            ids = jax.lax.dynamic_slice(order, (start,), (cfg.minibatch_size,))
            chunk = jax.tree.map(lambda leaf: leaf[ids], batch)
            (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, chunk, cfg)
            grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)
            grads = clip_by_global_norm(grads, cfg.max_grad_norm)
            params, opt_state = adam_step(params, grads, opt_state, cfg)
            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(minibatch_body, (params, opt_state), jnp.arange(steps_per_epoch))
        return (params, opt_state), metrics

    keys = jax.random.split(key, cfg.epochs)
    (params, opt_state), metrics = jax.lax.scan(epoch_body, (params, opt_state), keys)
    return params, opt_state, jax.tree.map(jnp.mean, metrics)


# --------------------------------------------------------------------------- #
# the training loop
# --------------------------------------------------------------------------- #
def normalise_advantages(advantages: jax.Array) -> jax.Array:
    """Zero-mean, unit-scale advantages -- but never manufactured from float dust.

    A rollout where nothing happened has all-zero advantages, and dividing those by
    their own (denormal) standard deviation turns rounding error into unit-scale
    "advantages" that shove the policy around at random.  When the spread is below
    ``1e-6`` the scale is left at one, so zero stays zero.
    """
    spread = advantages.std()
    scale = jnp.where(spread > 1e-6, spread, 1.0)
    return (advantages - advantages.mean()) / scale


def make_train(env, cfg: PPOConfig, num_actions: int | None = None):
    """Return ``train(key) -> (metrics, params)`` for this environment.

    Everything inside is a ``lax.scan``; wrap the result in ``jax.jit`` to compile the
    entire run, or call it directly to execute the same maths eagerly.
    """
    num_actions = num_actions or env.action_size

    def reset_all(key):
        return jax.vmap(env.reset)(jax.random.split(key, cfg.num_envs))

    def act(params, state, key):
        logits, value = forward(params, state.obs, cfg)
        keys = jax.random.split(key, cfg.num_envs)
        action, log_prob = jax.vmap(sample_action)(keys, logits)
        return action, log_prob, value

    def collect(params, state, key):
        def body(carry, step_key):
            state, key = carry
            key, act_key = jax.random.split(key)
            action, log_prob, value = act(params, state, act_key)
            stepped = jax.vmap(env.step)(state, action)
            transition = {
                'obs': state.obs,
                'actions': action,
                'log_probs': log_prob,
                'values': value,
                'rewards': stepped.reward,
                'terminated': stepped.info['terminated'],
                'truncated': stepped.info['truncated'],
            }
            return (stepped, key), transition

        (state, key), transitions = jax.lax.scan(body, (state, key), jax.random.split(key, cfg.rollout_len))
        _, final_values = forward(params, state.info['final_obs'], cfg)
        return state, key, transitions, final_values

    def train(key):
        key, init_key = jax.random.split(key)
        params = init_params(init_key, cfg=cfg, num_actions=num_actions)
        opt_state = adam_init(params)
        state = reset_all(key)

        def update_body(carry, step_key):
            params, opt_state, state, key = carry
            key, collect_key, update_key = jax.random.split(key, 3)
            state, key, transitions, final_values = collect(params, state, collect_key)
            advantages, returns = compute_gae(
                transitions['rewards'],
                transitions['values'],
                transitions['terminated'],
                transitions['truncated'],
                final_values,
                cfg,
            )
            batch = {
                'obs': transitions['obs'].reshape(-1, *transitions['obs'].shape[2:]),
                'actions': transitions['actions'].reshape(-1),
                'log_probs': transitions['log_probs'].reshape(-1),
                'advantages': advantages.reshape(-1),
                'returns': returns.reshape(-1),
            }
            # normalised across the rollout, as in the standard recipe
            batch['advantages'] = normalise_advantages(batch['advantages'])
            params, opt_state, metrics = _update(params, opt_state, batch, update_key, cfg)
            metrics = {**metrics, 'episode_return': jnp.mean(transitions['rewards'])}
            del step_key
            return (params, opt_state, state, key), metrics

        keys = jax.random.split(key, cfg.total_updates)
        (params, opt_state, state, key), metrics = jax.lax.scan(update_body, (params, opt_state, state, key), keys)
        metrics['steps'] = jnp.asarray(cfg.total_updates * cfg.batch_size)
        return metrics, params

    return train


def train(env, cfg: PPOConfig, key: jax.Array, jit: bool = True):
    """Run PPO; ``jit=False`` executes the identical program without compilation."""
    runner = make_train(env, cfg)
    if jit:
        runner = jax.jit(runner)
    return runner(key)
