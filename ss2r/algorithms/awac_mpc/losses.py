"""Value-based MPO actor losses and TD critic losses for TreeMPC rollouts."""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params
from jax.scipy import optimize as jax_optimize

from ss2r.algorithms.awac_mpc.networks import AWACMPCNetworks

Transition: TypeAlias = types.Transition


def _reduce_value(value_estimates: jax.Array, use_bro: bool) -> jax.Array:
    if use_bro:
        return jnp.mean(value_estimates, axis=-1)
    return jnp.min(value_estimates, axis=-1)


def _logmeanexp(x: jax.Array, axis: int = -1) -> jax.Array:
    x_max = jnp.max(x, axis=axis, keepdims=True)
    lme = jnp.log(jnp.mean(jnp.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return jnp.squeeze(lme, axis=axis)


def _solve_eta_dual(
    advantages: jax.Array,
    mpo_eta_epsilon: float,
    mpo_eta_init: float,
    mpo_eta_opt_maxiter: int,
) -> jax.Array:
    advantages = jax.lax.stop_gradient(advantages)
    x0 = jnp.asarray([jnp.log(mpo_eta_init)], dtype=jnp.float32)

    def eta_dual(log_eta_vec: jax.Array) -> jax.Array:
        eta = jnp.exp(log_eta_vec[0]) + 1e-8
        scaled_advantages = advantages / eta
        lme = _logmeanexp(scaled_advantages, axis=-1)
        return eta * (mpo_eta_epsilon + jnp.mean(lme))

    result = jax_optimize.minimize(
        eta_dual,
        x0,
        method="BFGS",
        options={"maxiter": mpo_eta_opt_maxiter},
    )
    return jnp.exp(result.x[0]) + 1e-8


def make_losses(
    awac_network: AWACMPCNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    mpo_eta_init: float,
    mpo_eta_epsilon: float,
    mpo_eta_min: float,
    mpo_eta_opt_maxiter: int,
    mpo_log_prob_min: float,
    use_bro: bool,
):
    if mpo_eta_init <= 0.0:
        raise ValueError(f"mpo_eta_init must be > 0, got {mpo_eta_init}.")
    if mpo_eta_epsilon <= 0.0:
        raise ValueError(f"mpo_eta_epsilon must be > 0, got {mpo_eta_epsilon}.")
    if mpo_eta_min <= 0.0:
        raise ValueError(f"mpo_eta_min must be > 0, got {mpo_eta_min}.")
    if mpo_eta_opt_maxiter < 1:
        raise ValueError(
            f"mpo_eta_opt_maxiter must be >= 1, got {mpo_eta_opt_maxiter}."
        )
    if mpo_log_prob_min > 0.0:
        raise ValueError(f"mpo_log_prob_min must be <= 0, got {mpo_log_prob_min}.")

    policy_network = awac_network.policy_network
    value_network = awac_network.value_network
    parametric_action_distribution = awac_network.parametric_action_distribution

    def critic_loss(
        value_params: Params,
        normalizer_params: Any,
        target_value_params: Params,
        transitions: Transition,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        value_old = value_network.apply(
            normalizer_params,
            value_params,
            transitions.observation,
        )
        next_value = value_network.apply(
            normalizer_params,
            target_value_params,
            transitions.next_observation,
        )
        target_value = transitions.reward * reward_scaling + transitions.discount * (
            discounting * _reduce_value(next_value, use_bro)
        )
        value_error = value_old - jnp.expand_dims(target_value, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        value_error *= jnp.expand_dims(1 - truncation, -1)
        loss = 0.5 * jnp.mean(jnp.square(value_error))
        aux = {
            "value_data_mean": jnp.mean(value_old),
            "value_data_std": jnp.std(value_old),
            "value_target_mean": jnp.mean(target_value),
            "value_target_std": jnp.std(target_value),
            "td_error_mean": jnp.mean(value_error),
            "td_error_abs_mean": jnp.mean(jnp.abs(value_error)),
            "td_error_abs_max": jnp.max(jnp.abs(value_error)),
        }
        return loss, aux

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        policy_extras = transitions.extras["policy_extras"]
        candidate_raw_actions = policy_extras["candidate_raw_actions"]
        candidate_advantages = policy_extras["candidate_advantages"]

        eta = _solve_eta_dual(
            candidate_advantages,
            mpo_eta_epsilon=mpo_eta_epsilon,
            mpo_eta_init=mpo_eta_init,
            mpo_eta_opt_maxiter=mpo_eta_opt_maxiter,
        )
        eta = jnp.maximum(eta, mpo_eta_min)
        mpo_scores = candidate_advantages / eta
        mpo_scores = mpo_scores - jnp.max(mpo_scores, axis=-1, keepdims=True)
        mpo_weights = jax.nn.softmax(mpo_scores, axis=-1)
        mpo_weights = jax.lax.stop_gradient(mpo_weights)

        current_dist_params = policy_network.apply(
            normalizer_params,
            policy_params,
            transitions.observation,
        )

        def _log_prob_for_state(dist_params_b, raw_actions_na):
            return jax.vmap(
                lambda raw_action: parametric_action_distribution.log_prob(
                    dist_params_b, raw_action
                )
            )(raw_actions_na)

        sampled_log_probs_current = jax.vmap(_log_prob_for_state)(
            current_dist_params, candidate_raw_actions
        )
        finite_log_probs_current = jnp.nan_to_num(
            sampled_log_probs_current,
            nan=mpo_log_prob_min,
            neginf=mpo_log_prob_min,
            posinf=0.0,
        )
        clipped_log_probs_current = jnp.maximum(
            finite_log_probs_current,
            mpo_log_prob_min,
        )

        nll_loss_per_state = -jnp.sum(mpo_weights * clipped_log_probs_current, axis=-1)
        nll_loss = jnp.mean(nll_loss_per_state)
        aux = {
            "eta": eta,
            "nll_loss": nll_loss,
            "nll_loss_max": jnp.max(nll_loss_per_state),
            "weight_min": jnp.min(mpo_weights),
            "weight_max": jnp.max(mpo_weights),
            "weight_mean": jnp.mean(mpo_weights),
            "advantage_sample_mean": jnp.mean(candidate_advantages),
            "advantage_sample_std": jnp.std(candidate_advantages),
        }
        return nll_loss, aux

    return critic_loss, actor_loss
