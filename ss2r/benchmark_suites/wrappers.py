import jax
import jax.numpy as jp
from brax.envs import State, Wrapper


class ActionObservationDelayWrapper(Wrapper):
    """Wrapper for adding action and observation delays in Brax envs, using JAX."""

    def __init__(self, env, action_delay: int = 0, obs_delay: int = 0):
        super().__init__(env)
        self.action_delay = action_delay
        self.obs_delay = obs_delay

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # Initialize the action and observation buffers as part of the state.info
        action_buffer, obs_buffer = self._init_buffers(state)
        # Store buffers in the state info for later access
        state.info["action_buffer"] = action_buffer
        state.info["obs_buffer"] = obs_buffer
        return state

    def _init_buffers(self, state):
        # Initialize the action and observation buffers as part of the state.info
        zero_action = jp.zeros(self.env.action_size)
        action_buffer = jp.tile(zero_action[None], (self.action_delay + 1, 1))
        obs_buffer = jp.tile(state.obs[None], (self.obs_delay + 1, 1))
        # Store buffers in the state info for later access
        return action_buffer, obs_buffer

    def step(self, state: State, action: jax.Array) -> State:
        # Retrieve the buffers from the state info
        action_buffer = state.info["action_buffer"]
        obs_buffer = state.info["obs_buffer"]
        # Shift the buffers to add new action and observation (delayed behavior)
        new_action_buffer = jp.roll(action_buffer, shift=-1, axis=0)
        new_action_buffer = new_action_buffer.at[-1].set(action)
        delayed_action = new_action_buffer[0]
        # Step the environment using the delayed action
        state = self.env.step(state, delayed_action)
        # Shift the observation buffer and add the current observation
        new_obs_buffer = jp.roll(obs_buffer, shift=-1, axis=0)
        new_obs_buffer = new_obs_buffer.at[-1].set(state.obs)
        delayed_obs = new_obs_buffer[0]

        # Update state observation with the delayed observation
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        init_action, init_obs = self._init_buffers(state)
        new_obs_buffer = where_done(init_obs, new_obs_buffer)
        new_action_buffer = where_done(init_action, new_action_buffer)
        # Update the buffers in state.info and return the updated state
        state.info["action_buffer"] = new_action_buffer
        state.info["obs_buffer"] = new_obs_buffer
        state = state.replace(
            obs=delayed_obs,
        )
        return state


class FrameActionStack(Wrapper):
    """Wrapper that stacks both observations and actions in a rolling manner for Brax environments.

    This wrapper maintains a history of both observations and actions, allowing the agent to access
    temporal information. For the initial state, the observation buffer is filled with the initial
    observation, and the action buffer is filled with zeros.

    Args:
        env: The Brax environment to wrap
        num_stack: Number of frames to stack (applies to both observations and actions)
    """

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack

        # Modify observation space to account for stacked frames and actions
        # Note: In Brax, we don't explicitly define spaces like in Gymnasium
        # but we'll track the dimensions for clarity
        self.single_obs_shape = self.env.observation_size
        self.single_action_shape = self.env.action_size
        self.num_stack = num_stack

    @property
    def observation_size(self) -> int:
        return self.num_stack * (self.single_obs_shape + self.single_action_shape)

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment and initialize the frame and action stacks."""
        state = self.env.reset(rng)
        # Create initial observation stack (filled with initial observation)
        action_buffer, obs_buffer = self._init_buffers(state)
        state.info["action_stack"] = action_buffer
        state.info["obs_stack"] = obs_buffer
        # Create the stacked observation
        state = state.replace(
            obs=self._get_stacked_obs(obs_buffer, action_buffer),
        )
        return state

    def _init_buffers(self, state):
        # Initialize the action and observation buffers as part of the state.info
        zero_action = jp.zeros(self.single_action_shape)
        action_buffer = jp.tile(zero_action[None], (self.num_stack, 1))
        obs_buffer = jp.tile(state.obs[None], (self.num_stack, 1))
        # Store buffers in the state info for later access
        return action_buffer, obs_buffer

    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment and update the stacks."""
        # Get current stacks
        action_buffer = state.info["action_stack"]
        obs_buffer = state.info["obs_stack"]
        # Step the environment
        state = self.env.step(state, action)
        # Update observation stack
        new_obs_buffer = jp.roll(obs_buffer, shift=-1, axis=0)
        new_obs_buffer = new_obs_buffer.at[-1].set(state.obs)

        # Update action stack
        new_action_buffer = jp.roll(action_buffer, shift=-1, axis=0)
        new_action_buffer = new_action_buffer.at[-1].set(action)

        # Handle done states
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        # Create the stacked observation
        stacked_obs = self._get_stacked_obs(new_obs_buffer, new_action_buffer)
        init_action, init_obs = self._init_buffers(state)
        new_obs_buffer = where_done(init_obs, new_obs_buffer)
        new_action_buffer = where_done(init_action, new_action_buffer)
        # Update state
        state.info["action_stack"] = new_action_buffer
        state.info["obs_stack"] = new_obs_buffer
        state = state.replace(
            obs=stacked_obs,
        )
        return state

    def _get_stacked_obs(
        self, obs_stack: jax.Array, action_stack: jax.Array
    ) -> jax.Array:
        """Combine the observation and action stacks into a single observation."""
        # Flatten the observation stack
        flat_obs = obs_stack.reshape(-1)
        # Flatten the action stack
        flat_actions = action_stack.reshape(-1)
        # Concatenate them
        return jp.concatenate([flat_obs, flat_actions])
