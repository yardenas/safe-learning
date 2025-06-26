import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base

from ss2r.algorithms.sac.types import float32


class ModelBasedEnv(envs.Env):
    """Environment wrapper that uses a learned model for predictions."""

    def __init__(
        self,
        transitions,
        observation_size,
        action_size,
        mbpo_network,
        training_state,
        ensemble_selection="mean",
        safety_budget=float("inf"),
        cost_discount=1.0,
        scaling_fn=lambda x: x,
        use_termination=True,
        termination_prob=1,
    ):
        super().__init__()
        self.model_network = mbpo_network.model_network
        self.model_params = training_state.model_params
        self.qc_network = mbpo_network.qc_network
        self.qc_params = training_state.qc_params
        self.qr_network = mbpo_network.qr_network
        self.backup_qr_params = training_state.backup_qr_params
        self.policy_network = mbpo_network.policy_network
        self.backup_policy_params = training_state.backup_policy_params
        self.normalizer_params = training_state.normalizer_params
        self.ensemble_selection = ensemble_selection
        self.safety_budget = safety_budget
        self._observation_size = observation_size
        self._action_size = action_size
        self.transitions = transitions
        self.cost_discount = cost_discount
        self.scaling_fn = scaling_fn
        self.use_termination = use_termination
        self.termination_prob = termination_prob

    def reset(self, rng: jax.Array) -> base.State:
        sample_key, model_key = jax.random.split(rng)
        indcs = jax.random.randint(sample_key, (), 0, self.transitions.reward.shape[0])
        transitions = float32(
            jax.tree_util.tree_map(lambda x: x[indcs], self.transitions)
        )
        state = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=jnp.zeros_like(transitions.reward),
            info={
                "truncation": jnp.zeros_like(transitions.reward),
                "cost": transitions.extras["state_extras"].get(
                    "cost", jnp.zeros_like(transitions.reward)
                ),
                "key": model_key,
            },
        )
        return state

    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Predict next state, reward, and cost using the model
        key = state.info["key"]
        sample_key, key = jax.random.split(key)
        next_obs, reward, cost = _propagate_ensemble(
            self.model_network.apply,
            self.normalizer_params,
            self.model_params,
            state.obs,
            action,
            self.ensemble_selection,
            sample_key,
        )
        state.info["key"] = key
        done = jnp.zeros_like(reward, dtype=jnp.float32)
        truncation = jnp.zeros_like(reward, dtype=jnp.float32)
        state.info["cost"] = cost
        state.info["truncation"] = truncation
        if self.qc_network is not None:
            prev_cumulative_cost = state.obs["cumulative_cost"][0]
            curr_discount = state.obs["curr_discount"][0] * self.cost_discount
            expected_cost_for_traj = (
                self.scaling_fn(prev_cumulative_cost)
                + self.qc_network.apply(
                    self.normalizer_params,
                    self.qc_params,
                    state.obs,
                    action,
                ).mean(axis=-1)
                * curr_discount
            )
            done = jnp.where(
                (expected_cost_for_traj > self.safety_budget)
                & jax.random.bernoulli(
                    key, p=self.termination_prob, shape=expected_cost_for_traj.shape
                ),
                jnp.ones_like(done),
                done,
            )
            pred_backup_action = self.policy_network.apply
            backup_policy_params = self.backup_policy_params
            backup_action = pred_backup_action(
                self.normalizer_params, backup_policy_params, state.obs
            )[0]
            pred_qr = self.qr_network.apply
            backup_qr_params = self.backup_qr_params
            pessimistic_qr_pred = pred_qr(
                self.normalizer_params, backup_qr_params, state.obs, backup_action[None]
            ).mean(axis=-1)
            reward = jnp.where(
                (expected_cost_for_traj > self.safety_budget)
                & jax.random.bernoulli(
                    key, p=self.termination_prob, shape=expected_cost_for_traj.shape
                ),
                pessimistic_qr_pred,
                reward,
            )

            def reset_states(done, state, next_obs):
                """Reset the state if done."""
                key, reset_keys = jax.random.split(state.info["key"])
                state.info["key"] = key
                next_obs["cumulative_cost"] = (
                    state.obs["cumulative_cost"] + cost * curr_discount
                )
                reset_state_obs = self.reset(reset_keys).obs
                obs = jax.tree_map(
                    lambda x, y: jnp.where(done, x, y), reset_state_obs, next_obs
                )
                return state, obs

            state, next_obs = reset_states(done, state, next_obs)
        state = state.replace(
            obs=next_obs,
            reward=reward,
            done=done if self.use_termination else jnp.zeros_like(done),
            info=state.info,
        )
        return state

    def observation_size(self) -> int:
        """Return the size of the observation space."""
        return self._observation_size

    def action_size(self) -> int:
        """Return the size of the action space."""
        return self._action_size

    def backend(self) -> str:
        """Return the backend used by the environment."""
        return "model_based"


def create_model_env(
    transitions,
    mbpo_network,
    training_state,
    observation_size,
    action_size,
    ensemble_selection="random",
    safety_budget=float("inf"),
    cost_discount=1.0,
    scaling_fn=lambda x: x,
    use_termination=True,
    termination_prob=1.0,
) -> ModelBasedEnv:
    """Factory function to create a model-based environment."""
    return ModelBasedEnv(
        transitions=transitions,
        mbpo_network=mbpo_network,
        training_state=training_state,
        observation_size=observation_size,
        action_size=action_size,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget,
        cost_discount=cost_discount,
        scaling_fn=scaling_fn,
        use_termination=use_termination,
        termination_prob=termination_prob,
    )


def _propagate_ensemble(
    pred_fn,
    normalizer_params,
    model_params,
    obs,
    action,
    ensemble_selection,
    key,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate the ensemble predictions based on the selection method."""
    # Calculate the nominal predictions
    if ensemble_selection == "nominal":
        # Get the average model parameters
        avg_model_params = jax.tree_util.tree_map(
            lambda p: jnp.mean(p, axis=0), model_params
        )
        next_obs, reward, cost = pred_fn(
            normalizer_params, avg_model_params, obs, action
        )
    elif ensemble_selection == "random":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        # Randomly select one of the ensemble predictions
        idx = jax.random.randint(key, (1,), 0, reward_pred.shape[0])[0]
        next_obs = jax.tree_map(lambda x: x[idx], next_obs_pred)
        reward = reward_pred[idx]
        cost = cost_pred[idx]
    elif ensemble_selection == "mean":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        next_obs = jax.tree_map(lambda x: jnp.mean(x, axis=0), next_obs_pred)
        reward = jnp.mean(reward_pred, axis=0)
        cost = jnp.mean(cost_pred, axis=0)
    else:
        raise ValueError(f"Unknown ensemble selection: {ensemble_selection}")
    return next_obs, reward, cost
