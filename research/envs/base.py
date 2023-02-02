import gym
import numpy as np

# unused imports for TODO on parallel envs
# import cloudpickle
# import multiprocessing as mp
# from abc import ABC, abstractmethod
# from research.utils.trainer import get_env
# from research.utils.utils import np_dataset_alloc, get_from_batch, set_in_batch


def _get_space(low=None, high=None, shape=None, dtype=None):
    all_vars = [low, high, shape, dtype]
    if any([isinstance(v, dict) for v in all_vars]):
        all_keys = set()  # get all the keys
        for v in all_vars:
            if isinstance(v, dict):
                all_keys.update(v.keys())
        # Construct all the sets
        spaces = {}
        for k in all_keys:
            space_low = low.get(k, None) if isinstance(low, dict) else low
            space_high = high.get(k, None) if isinstance(high, dict) else high
            space_shape = shape.get(k, None) if isinstance(shape, dict) else shape
            space_type = dtype.get(k, None) if isinstance(dtype, dict) else dtype
            spaces[k] = _get_space(space_low, space_high, space_shape, space_type)
        # Construct the gym dict space
        return gym.spaces.Dict(**spaces)

    if shape is None and isinstance(high, int):
        assert low is None, "Tried to specify a discrete space with both high and low."
        return gym.spaces.Discrete(high)

    # Otherwise assume its a box.
    if low is None:
        low = -np.inf
    if high is None:
        high = np.inf
    if dtype is None:
        dtype = np.float32
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


class Empty(gym.Env):

    """
    An empty holder for defining supervised learning problems
    It works by specifying the ranges and shapes.
    """

    def __init__(
        self,
        observation_low=None,
        observation_high=None,
        observation_shape=None,
        observation_dtype=np.float32,
        observation_space=None,
        action_low=None,
        action_high=None,
        action_shape=None,
        action_dtype=np.float32,
        action_space=None,
    ):
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = _get_space(observation_low, observation_high, observation_shape, observation_dtype)
        if action_space is not None:
            self.action_space = action_space
        else:
            self.action_space = _get_space(action_low, action_high, action_shape, action_dtype)

    def step(self, action):
        raise NotImplementedError("Empty Env does not have step")

    def reset(self, **kwargs):
        raise NotImplementedError("Empty Env does not have reset")


'''
# Future code for vectorized environments.

def _env_worker(remote, parent_remote, env_args, auto_reset):
    """
    TODO: appropriately handle resets
    """
    parent_remote.close()
    env = get_env(*env_args)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)

class BaseVecEnv(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step_send(self, action):
        raise NotImplementedError

    @abstractmethod
    def step_recv(self, action):
        raise NotImplementedError

    def step(self, action):
        self.step_send(action)
        return self.step_recv()

    @abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        raise NotImplementedError

class DummyVecEnv(BaseVecEnv):

    def __init__(self, env, env_kwargs, wrapper, wrapper_kwargs, num_envs=1, auto_reset=False):
        super().__init__()
        self.envs = [get_env(env, env_kwargs, wrapper, wrapper_kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.max_episode_steps = self.envs[0]._max_episode_steps if hasattr(self.envs[0], "_max_episode_steps")
        self.num_envs = num_envs
        self.auto_reset = auto_reset

        # Setup the buffers
        self.obs_buffer = np_dataset_alloc(self.observation_space, self.num_envs)
        self.reward_buffer = np_dataset_alloc(0.0, self.num_envs)
        self.done_buffer = np_dataset_alloc(False, self.num_envs)
        self.discount_buffer = np_dataset_alloc(False, self.num_envs)

        # To get the info keys step the environment once
        action = self.action_space.sample()

        self.info_buffer = [dict() for _ in range(self.num_envs)]

    def step_send(self, actions):
        # Store the actions
        self.actions = actions

    def step_recv(self):
        # Apply the action
        for i, env in enumerate(self.envs):
            action = get_from_batch(self.actions, i)
            obs, reward, done, info = env.step(action)
            set_in_batch(self.obs_buffer, obs)
            set_in_batch(self.reward_buffer, reward)
            set_in_batch(self.done_buffer, done)
            # TODO: set the discount buffer
            self.info_buffer[i] = info

        # Return copys of everything
        return self.obs_buffer
'''
