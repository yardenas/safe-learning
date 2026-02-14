"""TD3 networks."""

from typing import Mapping, Protocol, Sequence, TypeVar

import flax
import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.types import PRNGKey
from flax import linen

from ss2r.algorithms.sac.networks import ActivationFn, make_q_network

NetworkType = TypeVar("NetworkType", covariant=True)


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        *,
        n_critics: int = 2,
        n_heads: int = 1,
        use_bro: bool = True,
    ) -> NetworkType:
        pass


@flax.struct.dataclass
class TD3Networks:
    policy_network: networks.FeedForwardNetwork
    qr_network: networks.FeedForwardNetwork


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_policy_network(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    class PolicyModule(linen.Module):
        @linen.compact
        def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
            x = obs
            for size in hidden_layer_sizes:
                x = linen.Dense(features=size)(x)
                x = activation(x)
            x = linen.Dense(features=action_size)(x)
            return jnp.tanh(x)

    policy_module = PolicyModule()

    def apply(processor_params, params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        return policy_module.apply(params, obs)

    obs_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_inference_fn(td3_networks: TD3Networks, exploration_noise: float = 0.0):
    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        policy_network = td3_networks.policy_network

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> tuple[types.Action, types.Extra]:
            param_subset = (params[0], params[1])
            actions = policy_network.apply(*param_subset, observations)
            if deterministic or exploration_noise == 0.0:
                return actions, {}
            noise = exploration_noise * jax.random.normal(key_sample, actions.shape)
            actions = jnp.clip(actions + noise, -1.0, 1.0)
            return actions, {}

        return policy

    return make_policy


def make_td3_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    value_obs_key: str = "state",
    policy_obs_key: str = "state",
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
) -> TD3Networks:
    policy_network = make_policy_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    qr_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    return TD3Networks(  # type: ignore
        policy_network=policy_network,
        qr_network=qr_network,
    )
