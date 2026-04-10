"""MPO actor losses and TD critic losses for tree-MPC rollouts."""

from typing import Any, TypeAlias

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey
from jax.scipy import optimize as jax_optimize

from ss2r.algorithms.sac.networks import SafeSACNetworks

Transition: TypeAlias = types.Transition

_POSITIVE_EPS = 1e-8
_DEFAULT_MIN_STD = 1e-3


@flax.struct.dataclass
class MPODualParams:
    soft_mean: jax.Array
    soft_std: jax.Array


def soft_value(value: jax.Array | float) -> jax.Array:
    """Converts a positive value into an inverse-softplus parameter."""
    value = jnp.asarray(value, dtype=jnp.float32)
    return jnp.where(value < 100.0, jnp.log(jnp.expm1(value)), value)


def positive_soft_value(value: jax.Array) -> jax.Array:
    return jax.nn.softplus(value) + _POSITIVE_EPS


def init_mpo_params(
    *,
    action_size: int,
    mpo_eta_init: float,
    mpo_dual_mean_init: float,
    mpo_dual_std_init: float,
    mpo_kl_per_dim: bool,
) -> tuple[jax.Array, MPODualParams]:
    dual_shape = (action_size,) if mpo_kl_per_dim else ()
    soft_eta = soft_value(mpo_eta_init)
    dual_params = MPODualParams(
        soft_mean=soft_value(jnp.full(dual_shape, mpo_dual_mean_init)),
        soft_std=soft_value(jnp.full(dual_shape, mpo_dual_std_init)),
    )
    return soft_eta, dual_params


def dual_values(dual_params: MPODualParams) -> tuple[jax.Array, jax.Array]:
    return (
        positive_soft_value(dual_params.soft_mean),
        positive_soft_value(dual_params.soft_std),
    )


def eta_value(soft_eta: jax.Array) -> jax.Array:
    return positive_soft_value(soft_eta)


def _reduce_q(q_values: jax.Array, use_bro: bool) -> jax.Array:
    if use_bro:
        return jnp.mean(q_values, axis=-1)
    return jnp.min(q_values, axis=-1)


def _logmeanexp(x: jax.Array, axis: int = -1) -> jax.Array:
    x_max = jnp.max(x, axis=axis, keepdims=True)
    lme = jnp.log(jnp.mean(jnp.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return jnp.squeeze(lme, axis=axis)


def _solve_eta_dual(
    q_values: jax.Array,
    mpo_eta_epsilon: float,
    soft_eta_init: jax.Array,
    mpo_eta_opt_maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    q_values = jax.lax.stop_gradient(q_values)
    x0 = jnp.asarray([soft_eta_init], dtype=jnp.float32)

    def eta_dual(soft_eta_vec: jax.Array) -> jax.Array:
        eta = positive_soft_value(soft_eta_vec[0])
        scaled_q = q_values / eta
        lme = _logmeanexp(scaled_q, axis=-1)
        return eta * (mpo_eta_epsilon + jnp.mean(lme))

    result = jax_optimize.minimize(
        eta_dual,
        x0,
        method="BFGS",
        options={"maxiter": mpo_eta_opt_maxiter},
    )
    next_soft_eta = jnp.where(
        jnp.isfinite(result.x[0]), result.x[0], jnp.asarray(soft_eta_init)
    )
    return eta_value(next_soft_eta), next_soft_eta


def _sample_raw_actions(
    dist_params: jax.Array,
    key: PRNGKey,
    num_action_samples: int,
    parametric_action_distribution,
) -> tuple[jax.Array, jax.Array]:
    sample_keys = jax.random.split(key, num_action_samples)
    raw_actions_nba = jax.vmap(
        lambda k: parametric_action_distribution.sample_no_postprocessing(
            dist_params, k
        )
    )(sample_keys)
    actions_nba = jax.vmap(parametric_action_distribution.postprocess)(raw_actions_nba)
    return raw_actions_nba, actions_nba


def _extract_gaussian_params(
    dist_params: jax.Array, min_std: float, var_scale: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    mean, raw_std = jnp.split(dist_params, 2, axis=-1)
    std = (jax.nn.softplus(raw_std) + min_std) * var_scale
    return mean, std


def _compute_split_kl(
    current_dist_params: jax.Array,
    target_dist_params: jax.Array,
    *,
    min_std: float,
    var_scale: float,
    mpo_kl_per_dim: bool,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    current_mean, current_std = _extract_gaussian_params(
        current_dist_params, min_std, var_scale
    )
    target_mean, target_std = _extract_gaussian_params(
        target_dist_params, min_std, var_scale
    )

    kl_mean_per_dim = (
        0.5 * jnp.square(current_mean - target_mean) / jnp.square(target_std)
    )
    kl_std_per_dim = (
        jnp.log(current_std / target_std)
        + 0.5 * jnp.square(target_std / current_std)
        - 0.5
    )

    if mpo_kl_per_dim:
        kl_mean_constraint = jnp.mean(kl_mean_per_dim, axis=0)
        kl_std_constraint = jnp.mean(kl_std_per_dim, axis=0)
    else:
        kl_mean_constraint = jnp.mean(jnp.sum(kl_mean_per_dim, axis=-1))
        kl_std_constraint = jnp.mean(jnp.sum(kl_std_per_dim, axis=-1))

    stats = {
        "kl_mean": jnp.mean(jnp.sum(kl_mean_per_dim, axis=-1)),
        "kl_std": jnp.mean(jnp.sum(kl_std_per_dim, axis=-1)),
        "kl_total": jnp.mean(jnp.sum(kl_mean_per_dim + kl_std_per_dim, axis=-1)),
    }
    return kl_mean_constraint, kl_std_constraint, stats


def make_losses(
    sac_network: SafeSACNetworks,
    *,
    reward_scaling: float,
    discounting: float,
    mpo_eta_epsilon: float,
    mpo_eta_opt_maxiter: int,
    mpo_num_action_samples: int,
    mpo_delta_M_mean: float,
    mpo_delta_M_std: float,
    mpo_kl_per_dim: bool,
    use_bro: bool,
):
    if mpo_eta_epsilon <= 0.0:
        raise ValueError(f"mpo_eta_epsilon must be > 0, got {mpo_eta_epsilon}.")
    if mpo_eta_opt_maxiter < 1:
        raise ValueError(
            f"mpo_eta_opt_maxiter must be >= 1, got {mpo_eta_opt_maxiter}."
        )
    if mpo_num_action_samples < 1:
        raise ValueError(
            f"mpo_num_action_samples must be >= 1, got {mpo_num_action_samples}."
        )
    if mpo_delta_M_mean <= 0.0:
        raise ValueError(f"mpo_delta_M_mean must be > 0, got {mpo_delta_M_mean}.")
    if mpo_delta_M_std <= 0.0:
        raise ValueError(f"mpo_delta_M_std must be > 0, got {mpo_delta_M_std}.")

    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    parametric_action_distribution = sac_network.parametric_action_distribution
    action_size = parametric_action_distribution.param_size // 2
    min_std = float(
        getattr(
            parametric_action_distribution,
            "_min_std",
            getattr(parametric_action_distribution, "min_std", _DEFAULT_MIN_STD),
        )
    )
    var_scale = float(
        getattr(
            parametric_action_distribution,
            "_var_scale",
            getattr(parametric_action_distribution, "var_scale", 1.0),
        )
    )
    kl_mean_epsilon = (
        mpo_delta_M_mean / action_size if mpo_kl_per_dim else mpo_delta_M_mean
    )
    kl_std_epsilon = (
        mpo_delta_M_std / action_size if mpo_kl_per_dim else mpo_delta_M_std
    )

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
        target_policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        dual_params: MPODualParams,
        soft_eta: jax.Array,
        transitions: Transition,
        key: PRNGKey,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        target_dist_params = policy_network.apply(
            normalizer_params, target_policy_params, transitions.observation
        )
        raw_actions_nba, actions_nba = _sample_raw_actions(
            target_dist_params,
            key,
            mpo_num_action_samples,
            parametric_action_distribution,
        )

        sampled_q_values = jax.vmap(
            lambda sampled_actions_ba: _reduce_q(
                qr_network.apply(
                    normalizer_params,
                    target_q_params,
                    transitions.observation,
                    sampled_actions_ba,
                ),
                use_bro,
            )
        )(actions_nba)
        # [N, B] -> [B, N]
        sampled_q_values = jnp.swapaxes(sampled_q_values, 0, 1)

        eta, next_soft_eta = _solve_eta_dual(
            sampled_q_values,
            mpo_eta_epsilon=mpo_eta_epsilon,
            soft_eta_init=soft_eta,
            mpo_eta_opt_maxiter=mpo_eta_opt_maxiter,
        )
        mpo_scores = sampled_q_values / eta
        mpo_scores = mpo_scores - jnp.max(mpo_scores, axis=-1, keepdims=True)
        mpo_weights = jax.nn.softmax(mpo_scores, axis=-1)
        mpo_weights = jax.lax.stop_gradient(mpo_weights)

        current_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        sampled_log_probs_current = jax.vmap(
            lambda raw_actions_ba: parametric_action_distribution.log_prob(
                current_dist_params, raw_actions_ba
            )
        )(raw_actions_nba)
        # [N, B] -> [B, N]
        sampled_log_probs_current = jnp.swapaxes(sampled_log_probs_current, 0, 1)

        nll_loss_per_state = -jnp.sum(mpo_weights * sampled_log_probs_current, axis=-1)
        nll_loss = jnp.mean(nll_loss_per_state)

        kl_mean_constraint, kl_std_constraint, kl_stats = _compute_split_kl(
            current_dist_params,
            target_dist_params,
            min_std=min_std,
            var_scale=var_scale,
            mpo_kl_per_dim=mpo_kl_per_dim,
        )
        dual_mean, dual_std = dual_values(dual_params)
        loss = (
            nll_loss
            + jnp.sum(dual_mean * kl_mean_constraint)
            + jnp.sum(dual_std * kl_std_constraint)
        )

        weight_entropy = -jnp.mean(
            jnp.sum(mpo_weights * jnp.log(mpo_weights + _POSITIVE_EPS), axis=-1)
        )
        aux = {
            "eta": eta,
            "soft_eta_next": jax.lax.stop_gradient(next_soft_eta),
            "nll_loss": nll_loss,
            "weight_entropy": weight_entropy,
            "weight_min": jnp.min(mpo_weights),
            "weight_max": jnp.max(mpo_weights),
            "weight_mean": jnp.mean(mpo_weights),
            "q_sample_mean": jnp.mean(sampled_q_values),
            "q_sample_std": jnp.std(sampled_q_values),
            "kl_mean": kl_stats["kl_mean"],
            "kl_std": kl_stats["kl_std"],
            "kl_total": kl_stats["kl_total"],
            "kl_mean_constraint": jax.lax.stop_gradient(kl_mean_constraint),
            "kl_std_constraint": jax.lax.stop_gradient(kl_std_constraint),
            "dual_mean": jnp.mean(dual_mean),
            "dual_std": jnp.mean(dual_std),
        }
        return loss, aux

    def dual_loss(
        dual_params: MPODualParams,
        kl_mean_constraint: jax.Array,
        kl_std_constraint: jax.Array,
    ) -> jax.Array:
        dual_mean, dual_std = dual_values(dual_params)
        return jnp.sum(
            dual_mean * (kl_mean_epsilon - jax.lax.stop_gradient(kl_mean_constraint))
        ) + jnp.sum(
            dual_std * (kl_std_epsilon - jax.lax.stop_gradient(kl_std_constraint))
        )

    return critic_loss, actor_loss, dual_loss
