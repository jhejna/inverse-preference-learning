import math
from typing import Dict, Optional

import gym
import numpy as np
import torch

from research.utils.utils import get_from_batch, np_dataset_alloc, set_in_batch

from .replay_buffer import ReplayBuffer


class FeedbackLabelDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discount: float = 0.99,
        nstep: int = 1,
        segment_size: int = 20,
        subsample_size: Optional[int] = None,
        batch_size: int = 64,
        capacity: int = 100000,
    ):
        super().__init__()
        self.discount = discount
        self.nstep = nstep
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.subsample_size = subsample_size
        self._capacity = capacity
        self._size = 0
        self._idx = 0

        # Construct the buffers
        self.obs_1_buffer = np_dataset_alloc(observation_space, self._capacity, begin_pad=(self.segment_size,))
        self.obs_2_buffer = np_dataset_alloc(observation_space, self._capacity, begin_pad=(self.segment_size,))
        self.action_1_buffer = np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
        self.action_2_buffer = np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
        self.label_buffer = np_dataset_alloc(0.5, self._capacity)

    def add(self, queries: Dict, labels: np.ndarray):
        assert all([labels.shape[0] == v.shape[0] for v in queries.values()])
        num_to_add = labels.shape[0]

        if self._idx + num_to_add > self._capacity:
            # We have more segments than capacity allows, complete in two writes.
            num_b4_wrap = self._capacity - self._idx
            self.add(get_from_batch(queries, 0, num_b4_wrap), labels[:num_b4_wrap])
            self.add(get_from_batch(queries, num_b4_wrap, num_to_add), labels[num_b4_wrap:])
        else:
            start, end = self._idx, self._idx + num_to_add
            set_in_batch(self.obs_1_buffer, queries["obs_1"], start, end)
            set_in_batch(self.obs_2_buffer, queries["obs_2"], start, end)
            set_in_batch(self.action_1_buffer, queries["action_1"], start, end)
            set_in_batch(self.action_2_buffer, queries["action_2"], start, end)
            self.label_buffer[start:end] = labels
            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def _sample(self, idxs):
        if self.subsample_size is None:
            obs_1 = self.obs_1_buffer[idxs]
            obs_2 = self.obs_2_buffer[idxs]
            action_1 = self.action_1_buffer[idxs]
            action_2 = self.action_2_buffer[idxs]
            label = self.label_buffer[idxs]
        else:
            start = np.random.randint(0, self.segment_size - self.subsample_size)
            end = start + self.subsample_size
            obs_1 = self.obs_1_buffer[idxs, start:end]
            obs_2 = self.obs_2_buffer[idxs, start:end]
            action_1 = self.action_1_buffer[idxs, start:end]
            action_2 = self.action_2_buffer[idxs, start:end]
            label = self.label_buffer[idxs]
        # Note: we don't need to sample the states for this
        return dict(obs_1=obs_1, obs_2=obs_2, action_1=action_1, action_2=action_2, label=label)

    def __len__(self):
        return self._size

    def __iter__(self):
        assert torch.utils.data.get_worker_info() is None, "FeedbackLabel Dataset is not designed for parallelism."
        idxs = np.random.permutation(len(self))
        for i in range(math.ceil(len(self) / self.batch_size)):  # Need to use ceil to get all data points.
            cur_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]
            yield self._sample(cur_idxs)
