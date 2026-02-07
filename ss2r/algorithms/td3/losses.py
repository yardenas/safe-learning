"""TD3 losses.

See: https://arxiv.org/pdf/1802.09477.pdf
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.td3.networks import TD3Networks

Transition: TypeAlias = types.Transition


def make_losses(
    td3_network: TD3Networks,
    discounting: float,
    reward_scaling: float,
    policy_noise: float,
    noise_clip: float,
    n_critics: int,
    n_heads: int,
):
    policy_network = td3_network.policy_network
    qr_network = td3_network.qr_network

    def reshape_q_values(q_values: jnp.ndarray) -> jnp.ndarray:
        q_values = q_values.reshape(q_values.shape[0], n_critics, n_heads)
        return jnp.mean(q_values, axis=-1)

    def critic_loss(
        q_params: Params,
        target_q_params: Params,
        target_policy_params: Params,
        normalizer_params: types.Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        next_action = policy_network.apply(
            normalizer_params, target_policy_params, transitions.next_observation
        )
        noise = policy_noise * jax.random.normal(key, next_action.shape)
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        next_action = jnp.clip(next_action + noise, -1.0, 1.0)
        target_q_values = qr_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        target_q_values = reshape_q_values(target_q_values)
        target_q = jnp.min(target_q_values, axis=-1)
        target = (
            reward_scaling * transitions.reward
            + discounting * transitions.discount * target_q
        )
        truncation = transitions.extras["state_extras"]["truncation"]
        q_values = qr_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        q_values = reshape_q_values(q_values)
        q_error = q_values - target[:, None]
        q_error *= (1 - truncation)[:, None]
        return 0.5 * jnp.mean(jnp.square(q_error))

    def actor_loss(
        policy_params: Params,
        normalizer_params: types.Params,
        q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        del key
        action = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        q_values = qr_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        q_values = reshape_q_values(q_values)
        q1 = q_values[:, 0]
        return -jnp.mean(q1)

    return critic_loss, actor_loss
