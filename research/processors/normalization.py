from typing import Dict, List, Union

import gym
import numpy as np
import torch

from research.utils import utils

from .base import Processor


class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, epsilon: float = 1e-6):
        super().__init__()
        self.shape = shape
        self._mean = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self._var = torch.nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self._count = torch.nn.Parameter(torch.tensor(epsilon, dtype=torch.float), requires_grad=False)

    def update(self, x: Union[float, np.ndarray, torch.Tensor]) -> None:
        if isinstance(x, float):
            x = torch.tensor(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Invalid type provided")
        # If the data is unbatched, unsqueeze it
        if len(x.shape) == len(self.shape):
            x = x.unsqueeze(0)

        mean = torch.mean(x, dim=0)
        var = torch.var(x, dim=0, unbiased=False)
        count = x.shape[0]

        delta = mean - self._mean
        total_count = self._count + count

        new_mean = self._mean + delta * count / total_count

        m_a = self._var * self._count
        m_b = var * count
        m_2 = m_a + m_b + torch.square(delta) * self._count * count / total_count
        new_var = m_2 / total_count
        # Update member variables
        self._count.copy_(total_count)
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def std(self):
        return torch.sqrt(self._var)


class RunningObservationNormalizer(Processor):
    """
    A running observation normalizer.
    Note that there are quite a few speed optimizations that could be performed:
    1. We could cache computation of the variance etc. so it doesn't run everytime.
    2. We could permanently store torch tensors so we don't recompute them and sync to GPU.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        epsilon: float = 1e-7,
        clip: float = 10,
        explicit_update: bool = False,
        paired_keys: List[str] = [],
    ) -> None:
        super().__init__(observation_space, action_space)
        if isinstance(observation_space, gym.spaces.Dict):
            assert all([isinstance(space, gym.spaces.Box) for space in observation_space.values()])
            self.rms = {
                k: RunningMeanStd(space.shape, epsilon=epsilon)
                for k, space in observation_space.items()
                if k not in paired_keys
            }
            if len(paired_keys) > 0:
                self.rms["paired"] = RunningMeanStd(observation_space[paired_keys[0]].shape, epsilon=epsilon)
            self.paired_keys = set(paired_keys)
        elif isinstance(observation_space, gym.spaces.Box):
            self.rms = RunningMeanStd(observation_space.shape, epsilon=epsilon)
        else:
            raise ValueError("Invalid space type provided.")
        self._updated_stats = True
        self.clip = clip
        self.explicit_update = explicit_update

    @property
    def supports_gpu(self):
        return False

    def _get_key(self, k):
        if k in self.paired_keys:
            return "paired"
        else:
            return k

    def update(self, obs: Union[torch.Tensor, Dict]) -> None:
        if isinstance(obs, dict):
            for k in obs.keys():
                self.rms[self._get_key(k)].update(obs[k])
        else:
            self.rms.update(obs)
        self._updated_stats = True

    def normalize(self, obs: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict]:
        if self._updated_stats:
            # Grab the states from the RMS trackers
            self._mean = {k: self.rms[k].mean for k in self.rms.keys()} if isinstance(obs, dict) else self.rms.mean
            self._std = {k: self.rms[k].std for k in self.rms.keys()} if isinstance(obs, dict) else self.rms.std
            device = utils.get_device(obs)
            if device is not None:
                self._mean = utils.to_device(self._mean, device)
                self._std = utils.to_device(self._std, device)
            self._updated_stats = False
        # Normalize the observation
        if isinstance(obs, dict):
            obs = {k: (obs[k] - self._mean[self._get_key(k)]) / self._std[self._get_key(k)] for k in obs.keys()}
            if self.clip is not None:
                for k in obs.keys():
                    obs[k] = torch.clamp(obs[k], -self.clip, self.clip)
            return obs
        elif isinstance(obs, torch.Tensor):
            obs = (obs - self._mean) / self._std
            return obs if self.clip is None else torch.clamp(obs, -self.clip, self.clip)
        else:
            raise ValueError("Invalid Input provided")

    def forward(self, batch):
        # Check if we should update the statistics
        if not self.explicit_update and self.training and "obs" in batch:
            self.update(batch["obs"])
        # Normalize
        for k in ("obs", "next_obs", "init_obs"):
            if k in batch:
                batch[k] = self.normalize(batch[k])
        return batch
