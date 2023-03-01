from abc import abstractmethod

import gym
import numpy as np

from research.datasets.replay_buffer import ReplayBuffer
from research.envs.base import Empty

from .base import Algorithm


class OffPolicyAlgorithm(Algorithm):
    def __init__(
        self,
        *args,
        offline_steps: int = 0,  # Run fully offline by setting to -1
        random_steps: int = 1000,
        env_freq: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert issubclass(self.dataset_class, ReplayBuffer)
        self.offline_steps = offline_steps
        self.env_freq = env_freq
        self.random_steps = random_steps

    def setup(self):
        if isinstance(self.env, Empty):
            return
        else:
            # We have an environment, setup everything.
            self._current_obs = self.env.reset()
            self._episode_reward = 0
            self._episode_length = 0
            self._num_ep = 0
            self._env_steps = 0
            # Note that currently the very first (s, a) pair is thrown away because
            # we don't add to the dataset here.
            # This was done for better compatibility for offline to online learning.

    @abstractmethod
    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        raise NotImplementedError

    def train_extras(self, step: int, total_steps: int):
        # Return if env is Empty or we we aren't at every env_freq steps
        metrics = dict()
        if isinstance(self.env, Empty) or step % self.env_freq != 0 or self.offline_steps < 0:
            return metrics
        elif self.offline_steps >= 0 and step < self.offline_steps:
            metrics["env_steps"] = self._env_steps
            metrics["reward"] = -np.inf  # purposefullly set nan so we don't rewrite csv log
            metrics["length"] = np.inf  # purposefullly set nan so we don't rewrite csv log
            metrics["num_ep"] = self._num_ep
            return metrics

        if self._env_steps < self.random_steps:
            action = self.env.action_space.sample()
        else:
            self.eval()
            action = self._get_train_action(step, total_steps)
            self.train()
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(next_obs, action, reward, done, discount)

        if done:
            self._num_ep += 1
            # update metrics
            metrics["reward"] = self._episode_reward
            metrics["length"] = self._episode_length
            metrics["num_ep"] = self._num_ep
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics["env_steps"] = self._env_steps
        return metrics
