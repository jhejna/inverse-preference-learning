import math
import os
from typing import Dict, List

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
        batch_size: int = 64,
        capacity: int = 100000,
    ):
        super().__init__()
        self.discount = discount
        self.nstep = nstep
        self.batch_size = batch_size
        self.segment_size = segment_size
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
        obs_1 = self.obs_1_buffer[idxs]
        obs_2 = self.obs_2_buffer[idxs]
        action_1 = self.action_1_buffer[idxs]
        action_2 = self.action_2_buffer[idxs]
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


class MultiTaskOracleFeedbackDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        paths: List[str],
        capacity: int = 10000,
        segment_size: int = 25,
        **kwargs,
    ):
        super().__init__()
        if isinstance(paths, str):
            paths = [os.path.join(paths, p) for p in os.listdir(paths)]

        # Get the basenames of the directories to use as task names
        task_names = [os.path.basename(p) for p in paths]
        self._task_to_id = {task: index for index, task in enumerate(task_names)}
        self._id_to_task = {v: k for k, v in self._task_to_id.items()}
        self._datasets = {}
        for task, p in zip(task_names, paths):
            replay_buffer = ReplayBuffer(observation_space, action_space, path=p, distributed=False, **kwargs)
            dataset = FeedbackLabelDataset(
                observation_space, action_space, capacity=capacity, segment_size=segment_size, **kwargs
            )
            # Sample segments from the replay buffer like in PEBBLE
            batch = replay_buffer.sample(batch_size=2 * capacity, stack=segment_size, pad=0)
            # Compute the discounted reward across each segment to be used for oracle labels
            returns = np.sum(
                batch["reward"] * np.power(replay_buffer.discount, np.arange(batch["reward"].shape[1])), axis=1
            )
            queries = dict(
                obs_1=batch["obs"][:capacity],
                obs_2=batch["obs"][capacity:],
                action_1=batch["action"][:capacity],
                action_2=batch["action"][capacity:],
            )
            labels = 1.0 * (returns[:capacity] < returns[capacity:])
            dataset.add(queries, labels)
            # Explicitly delete the replay buffer
            del replay_buffer
            self._datasets[task] = dataset
            print("Finished", task)

    def __iter__(self):
        assert torch.utils.data.get_worker_info() is None, "FeedbackLabel Dataset is not designed for parallelism."
        iterators = {task: iter(dataset) for task, dataset in self._datasets.items()}
        while True:
            for task, iterator in iterators.items():
                try:
                    batch = next(iterator)
                except StopIteration:
                    # We ran out of batches in one of the datasets. This is approximately one epoch, so return.
                    return
                yield (self._task_to_id[task], batch)
