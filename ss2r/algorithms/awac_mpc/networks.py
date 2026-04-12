"""Networks for value-based AWAC-MPC."""

from typing import Mapping, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.sac.networks import MLP, BroNet


@flax.struct.dataclass
class AWACMPCNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_value_network(
    observation_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    obs_key: str = "state",
    use_bro: bool = True,
    n_critics: int = 2,
) -> networks.FeedForwardNetwork:
    """Creates an ensemble state-value network."""

    class ValueModule(linen.Module):
        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            outputs = []
            net = BroNet if use_bro else MLP
            for _ in range(self.n_critics):
                value = net(  # type: ignore
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(obs)
                outputs.append(value)
            return jnp.concatenate(outputs, axis=-1)

    value_module = ValueModule(n_critics=n_critics)

    def apply(processor_params, value_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        return value_module.apply(value_params, obs)

    obs_size = _get_obs_state_size(observation_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs),
        apply=apply,
    )


def make_awac_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    value_obs_key: str = "state",
    policy_obs_key: str = "state",
    use_bro: bool = True,
    n_critics: int = 2,
) -> AWACMPCNetworks:
    """Creates policy and ensemble value networks for AWAC-MPC."""

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        use_bro=use_bro,
        n_critics=n_critics,
    )
    return AWACMPCNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
