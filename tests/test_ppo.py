"""The PPO baseline: shapes, advantage estimation and one step of each code path.

The interesting invariant here is that the jitted and eager runs are the *same*
program -- ``examples/train_ppo.py --compare`` relies on that, and it only holds if
every random draw comes from the same key in the same order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from procgen_jax import MazeEnv, make_env
from procgen_jax.baselines import (
    PPOConfig,
    compute_gae,
    forward,
    greedy_action,
    init_params,
    make_train,
    normalise_advantages,
    ppo_loss,
    sample_action,
)

ENV = MazeEnv(world_dim=15)
SMALL = PPOConfig(num_envs=2, rollout_len=8, total_updates=2, num_minibatches=2)


def test_forward_and_sampling_shapes():
    params = init_params(jax.random.PRNGKey(0), cfg=SMALL, num_actions=ENV.action_size)
    obs = ENV.reset(jax.random.PRNGKey(0)).obs

    logits, value = forward(params, obs, SMALL)
    assert logits.shape == (ENV.action_size,)
    assert value.shape == ()
    assert np.isfinite(np.asarray(logits)).all()

    action, log_prob = sample_action(jax.random.PRNGKey(1), logits)
    assert 0 <= int(action) < ENV.action_size
    assert np.isfinite(float(log_prob))
    # the sampled log-prob has to match the categorical distribution it came from
    expected = jax.nn.log_softmax(logits)[action]
    assert float(log_prob) == pytest.approx(float(expected), abs=1e-6)


def test_pooling_matches_a_manual_mean():
    from procgen_jax.baselines.ppo import _features

    obs = jnp.arange(64 * 64 * 3, dtype=jnp.uint8).reshape(64, 64, 3)
    features = _features(obs, 4)

    assert features.shape == (16 * 16 * 3,)
    block = obs[:4, :4, 0].astype(jnp.float32) / 255.0
    assert float(features[0]) == pytest.approx(float(block.mean()), rel=1e-5)


def test_init_is_deterministic_and_moves_with_the_key():
    first = init_params(jax.random.PRNGKey(0), cfg=SMALL, num_actions=9)
    again = init_params(jax.random.PRNGKey(0), cfg=SMALL, num_actions=9)
    other = init_params(jax.random.PRNGKey(1), cfg=SMALL, num_actions=9)

    assert np.array_equal(np.asarray(first['w0']), np.asarray(again['w0']))
    assert not np.array_equal(np.asarray(first['w0']), np.asarray(other['w0']))


def test_gae_without_reward_is_zero():
    rewards = jnp.zeros((8, 2))
    values = jnp.zeros((8, 2))
    masks = jnp.zeros((8, 2), bool)
    advantages, returns = compute_gae(rewards, values, masks, masks, jnp.zeros((2,)), SMALL)

    assert advantages.shape == (8, 2)
    assert float(jnp.abs(advantages).max()) == 0.0
    assert float(jnp.abs(returns).max()) == 0.0


def test_gae_bootstraps_from_the_final_observation_on_truncation():
    """One truncated step: reward 0, value 1, so delta = gamma * 1 - 1."""
    rewards = jnp.zeros((1, 2))
    values = jnp.ones((1, 2))
    terminated = jnp.zeros((1, 2), bool)
    truncated = jnp.ones((1, 2), bool)

    advantages, returns = compute_gae(rewards, values, terminated, truncated, jnp.ones((2,)), SMALL)

    assert float(advantages[0, 0]) == pytest.approx(SMALL.gamma - 1.0, rel=1e-5)
    assert float(returns[0, 0]) == pytest.approx(SMALL.gamma, rel=1e-5)


def test_gae_cuts_the_trace_on_termination():
    """A real termination must not bootstrap, whatever the final value says."""
    rewards = jnp.zeros((1, 2))
    values = jnp.ones((1, 2))
    terminated = jnp.ones((1, 2), bool)
    truncated = jnp.zeros((1, 2), bool)

    advantages, returns = compute_gae(rewards, values, terminated, truncated, jnp.ones((2,)) * 100, SMALL)

    assert float(advantages[0, 0]) == pytest.approx(-1.0, rel=1e-5)
    assert float(returns[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_gae_matches_a_reference_loop():
    """Compare against the textbook loop, written out over numpy."""
    length, envs = 6, 2
    key = jax.random.PRNGKey(0)
    rewards = jax.random.normal(key, (length, envs))
    values = jax.random.normal(jax.random.PRNGKey(1), (length, envs))
    terminated = jnp.zeros((length, envs), bool).at[3, 0].set(True)
    truncated = jnp.zeros((length, envs), bool).at[5, 1].set(True)
    final_values = jnp.asarray([0.3, -0.7])

    advantages, returns = compute_gae(rewards, values, terminated, truncated, final_values, SMALL)

    r, v = np.asarray(rewards), np.asarray(values)
    done = np.asarray(terminated), np.asarray(truncated)
    expected = np.zeros((length, envs))
    running = np.zeros(envs)
    for step in reversed(range(length)):
        next_value = np.asarray(final_values) if step == length - 1 else v[step + 1]
        for env in range(envs):
            alive = 1.0 - float(done[0][step, env] or done[1][step, env])
            keep = 1.0 - float(done[0][step, env])
            delta = r[step, env] + SMALL.gamma * next_value[env] * keep - v[step, env]
            running[env] = delta + SMALL.gamma * SMALL.gae_lambda * alive * running[env]
            expected[step, env] = running[env]

    np.testing.assert_allclose(np.asarray(advantages), expected, rtol=1e-5, atol=1e-6)


def test_ppo_loss_is_finite_and_penalises_bad_actions():
    params = init_params(jax.random.PRNGKey(0), cfg=SMALL, num_actions=4)
    obs = jnp.zeros((4, 64, 64, 3), jnp.uint8)
    logits, _ = forward(params, obs, SMALL)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = jnp.zeros((4,), jnp.int32)

    batch = {
        'obs': obs,
        'actions': actions,
        'log_probs': jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0],
        'advantages': jnp.ones((4,)),
        'returns': jnp.zeros((4,)),
    }
    loss, metrics = ppo_loss(params, batch, SMALL)
    assert np.isfinite(float(loss))
    assert set(metrics) == {'loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'}


def test_one_update_changes_the_parameters():
    runner = jax.jit(make_train(ENV, SMALL))
    # the loop derives its init key as split(key)[1], so this is where training starts
    _, init_key = jax.random.split(jax.random.PRNGKey(1))
    params = init_params(init_key, cfg=SMALL, num_actions=ENV.action_size)

    metrics, trained = runner(jax.random.PRNGKey(1))

    assert metrics['loss'].shape == (SMALL.total_updates,)
    assert np.isfinite(np.asarray(metrics['loss'])).all()
    assert not np.array_equal(np.asarray(params['w0']), np.asarray(trained['w0']))


def test_jit_and_eager_agree():
    """``--compare`` is only honest if compilation does not change the numbers."""
    cfg = PPOConfig(num_envs=2, rollout_len=8, total_updates=1, num_minibatches=2)
    runner = make_train(ENV, cfg)
    key = jax.random.PRNGKey(3)

    eager_metrics, eager_params = runner(key)
    jit_metrics, jit_params = jax.jit(make_train(ENV, cfg))(key)

    np.testing.assert_allclose(np.asarray(eager_metrics['loss']), np.asarray(jit_metrics['loss']), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(eager_params['w0']), np.asarray(jit_params['w0']), rtol=1e-5, atol=1e-5)


def test_greedy_action_is_the_argmax():
    params = init_params(jax.random.PRNGKey(0), cfg=SMALL, num_actions=ENV.action_size)
    obs = ENV.reset(jax.random.PRNGKey(0)).obs
    logits, _ = forward(params, obs, SMALL)

    assert int(greedy_action(params, obs, SMALL)) == int(jnp.argmax(logits))


@pytest.mark.parametrize('game', ['maze', 'coinrun'])
def test_make_env_returns_a_usable_env(game):
    env = make_env(game)

    assert env.action_size > 0
    assert env.observation_shape == (64, 64, 3)


def test_normalise_advantages_keeps_zero_zero():
    """The guard that stops a silent rollout from being trained on rounding error."""
    zero = normalise_advantages(jnp.zeros(64))
    assert float(jnp.abs(zero).max()) == 0.0

    dust = jnp.linspace(-1.0, 1.0, 64) * 1e-9
    assert float(jnp.abs(normalise_advantages(dust)).max()) < 1e-6

    signal = jnp.asarray([0.0, 0.0, 10.0, 0.0])
    normalised = normalise_advantages(signal)
    assert float(normalised.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalised.std()) == pytest.approx(1.0, rel=1e-5)
    assert float(normalised[2]) > 0.0


def test_a_real_rollout_does_move_the_policy():
    cfg = PPOConfig(num_envs=2, rollout_len=8, total_updates=1, num_minibatches=2)
    _, init_key = jax.random.split(jax.random.PRNGKey(1))
    before = init_params(init_key, cfg=cfg, num_actions=ENV.action_size)

    _, trained = jax.jit(make_train(ENV, cfg))(jax.random.PRNGKey(1))

    assert float(jnp.abs(trained['policy_w'] - before['policy_w']).max()) > 0.0
