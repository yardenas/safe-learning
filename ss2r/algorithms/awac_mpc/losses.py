"""MPO-weighted actor losses and TD critic losses for tree-MPC rollouts."""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.networks import SafeSACNetworks

Transition: TypeAlias = types.Transition


def atanh_with_slack(y: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Stable inverse tanh used before log_prob evaluation."""
    y = y.astype(jnp.float32)
    eps_arr = jnp.asarray(eps, dtype=jnp.float32)
    y = jnp.clip(y, -1.0 + eps_arr, 1.0 - eps_arr)
    return 0.5 * (jnp.log1p(y) - jnp.log1p(-y))


def _reduce_q(q_values: jax.Array, use_bro: bool) -> jax.Array:
    if use_bro:
        return jnp.mean(q_values, axis=-1)
    return jnp.min(q_values, axis=-1)


def make_losses(
    sac_network: SafeSACNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    mpo_eta: float,
    mpo_num_action_samples: int,
    use_bro: bool,
):
    if mpo_eta <= 0.0:
        raise ValueError(f"mpo_eta must be > 0, got {mpo_eta}.")
    if mpo_num_action_samples < 1:
        raise ValueError(
            f"mpo_num_action_samples must be >= 1, got {mpo_num_action_samples}."
        )

    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        action = transitions.action
        q_old_action = qr_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        key, next_key = jax.random.split(key)
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample(next_dist_params, next_key)
        next_q = qr_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        next_v = _reduce_q(next_q, use_bro)
        target_q = transitions.reward * reward_scaling + transitions.discount * (
            discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)
        loss = 0.5 * jnp.mean(jnp.square(q_error))
        aux = {
            "q_data_mean": jnp.mean(q_old_action),
            "q_data_std": jnp.std(q_old_action),
            "q_target_mean": jnp.mean(target_q),
            "q_target_std": jnp.std(target_q),
            "td_error_mean": jnp.mean(q_error),
            "td_error_abs_mean": jnp.mean(jnp.abs(q_error)),
            "td_error_abs_max": jnp.max(jnp.abs(q_error)),
        }
        return loss, aux

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        sample_keys = jax.random.split(key, mpo_num_action_samples)

        sampled_actions = jax.vmap(
            lambda sample_k: parametric_action_distribution.sample(
                dist_params, sample_k
            )
        )(sample_keys)
        sampled_log_probs = jax.vmap(
            lambda action: parametric_action_distribution.log_prob(
                dist_params, atanh_with_slack(action)
            )
        )(sampled_actions)
        sampled_q_values = jax.vmap(
            lambda sampled_action: _reduce_q(
                qr_network.apply(
                    normalizer_params, q_params, transitions.observation, sampled_action
                ),
                use_bro,
            )
        )(sampled_actions)
        # [num_samples, batch] -> [batch, num_samples]
        sampled_log_probs = jnp.swapaxes(sampled_log_probs, 0, 1)
        sampled_q_values = jnp.swapaxes(sampled_q_values, 0, 1)

        mpo_scores = sampled_q_values / mpo_eta
        mpo_scores = mpo_scores - jnp.max(mpo_scores, axis=-1, keepdims=True)
        mpo_weights = jax.nn.softmax(mpo_scores, axis=-1)
        mpo_weights = jax.lax.stop_gradient(mpo_weights)
        loss_per_state = -jnp.sum(mpo_weights * sampled_log_probs, axis=-1)
        truncation = transitions.extras["state_extras"]["truncation"]
        loss_mask = 1.0 - truncation
        loss = jnp.mean(loss_per_state * loss_mask)
        return loss, {}

    return critic_loss, actor_loss
