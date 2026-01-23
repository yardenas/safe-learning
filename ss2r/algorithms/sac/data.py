import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey, Transition

from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.types import MakePolicyFn, UnrollFn
from ss2r.rl.utils import quantize_images, remove_pixels


def _add_env_state(transition: Transition, env_state: envs.State) -> Transition:
    extras = transition.extras or {}
    state_extras = dict(extras.get("state_extras", {}))
    state_extras["env_state"] = env_state
    new_extras = dict(extras)
    new_extras["state_extras"] = state_extras
    return transition._replace(extras=new_extras)


def get_collection_fn(cfg):
    store_env_state = bool(getattr(cfg.agent, "store_env_state", False))
    if cfg.agent.data_collection.name == "step":
        if store_env_state:
            return make_collection_fn(actor_step_with_state)
        return collect_single_step
    elif cfg.agent.data_collection.name == "episodic":

        def generate_episodic_unroll(
            env,
            env_state,
            make_policy_fn,
            policy_params,
            key,
            extra_fields,
        ):
            if store_env_state:
                env_state, transitions = generate_unroll_with_state(
                    env,
                    env_state,
                    make_policy_fn,
                    policy_params,
                    key,
                    cfg.training.episode_length,
                    extra_fields,
                )
            else:
                env_state, transitions = acting.generate_unroll(
                    env,
                    env_state,
                    make_policy_fn(policy_params),
                    key,
                    cfg.training.episode_length,
                    extra_fields,
                )
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
            return env_state, transitions

        return make_collection_fn(generate_episodic_unroll)
    elif cfg.agent.data_collection.name == "hardware":
        if store_env_state:
            raise ValueError(
                "store_env_state is not supported for hardware data collection."
            )
        data_collection_cfg = cfg.agent.data_collection
        if "Go1" in cfg.environment.task_name or "Go2" in cfg.environment.task_name:
            from ss2r.algorithms.sac.go1_sac_to_onnx import (
                go1_postprocess_data,
                make_go1_policy,
            )
            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = functools.partial(make_go1_policy, cfg=cfg)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                go1_postprocess_data,
                data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        elif "rccar" in cfg.environment.task_name:
            import cloudpickle

            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = lambda _, params: cloudpickle.dumps(params)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                lambda data, _: Transition(*data),
                address=data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        elif "PandaPickCubeCartesian" in cfg.environment.task_name:
            from ss2r.algorithms.sac.franka_sac_to_onnx import (
                make_franka_policy,
                postprocess_data,
            )
            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = functools.partial(make_franka_policy, cfg=cfg)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                postprocess_data,
                address=data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        else:
            raise ValueError(
                f"Environment {cfg.environment.task_name} not supported for hardware data collection."
            )
    else:
        raise ValueError(f"Unknown data collection {cfg.agent.data_collection.name}")


def actor_step(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.actor_step(env, env_state, policy, key, extra_fields)


def actor_step_with_state(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    new_env_state, transition = acting.actor_step(
        env, env_state, policy, key, extra_fields
    )
    transition = _add_env_state(transition, env_state)
    return new_env_state, transition


def generate_unroll(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    unroll_length,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.generate_unroll(
        env, env_state, policy, key, unroll_length, extra_fields
    )


def generate_unroll_with_state(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    unroll_length,
    extra_fields,
):
    policy = make_policy_fn(policy_params)

    def step_fn(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        new_state, transition = acting.actor_step(
            env, state, policy, subkey, extra_fields
        )
        transition = _add_env_state(transition, state)
        return (new_state, key), transition

    (env_state, _), transitions = jax.lax.scan(
        step_fn, (env_state, key), (), length=unroll_length
    )
    return env_state, transitions


def make_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn,
        params: Params,
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env,
            env_state,
            make_policy_fn,
            (normalizer_params, params),
            key,
            extra_fields=extra_fields,
        )
        normalizer_params = running_statistics.update(
            normalizer_params, remove_pixels(transitions.observation)
        )
        transitions = float16(transitions)
        transitions = transitions._replace(
            observation=quantize_images(transitions.observation),
            next_observation=quantize_images(transitions.next_observation),
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(actor_step)
