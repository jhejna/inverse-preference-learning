import io
import math
from typing import Dict, Optional

import gym
import numpy as np
import torch

from research.utils import utils

from .replay_buffer import ReplayBuffer, get_buffer_bytes


class PairwiseComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        discount: float = 0.99,
        nstep: int = 1,
        segment_size: int = 20,
        subsample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__()
        self.discount = discount
        self.nstep = nstep
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_size = segment_size
        self.subsample_size = subsample_size
        self._capacity = capacity

        if self._capacity is None:
            assert path is not None, "If capacity is not given, must have path to load from"
            with open(path, "rb") as f:
                data = np.load(f)
                data = utils.nest_dict(data)
            # Set the buffers to be the stored data. Woot woot.
            self.obs_1_buffer = data["obs_1"]
            self.obs_2_buffer = data["obs_2"]
            self.action_1_buffer = data["action_1"]
            self.action_2_buffer = data["action_2"]
            self.label_buffer = data["label"]
            self._size = len(self.label_buffer)
        else:
            # Construct the buffers
            self.obs_1_buffer = utils.np_dataset_alloc(
                observation_space, self._capacity, begin_pad=(self.segment_size,)
            )
            self.obs_2_buffer = utils.np_dataset_alloc(
                observation_space, self._capacity, begin_pad=(self.segment_size,)
            )
            self.action_1_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.action_2_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.label_buffer = utils.np_dataset_alloc(0.5, self._capacity)
            self._size = 0
            self._idx = 0
            if path is not None:
                assert path is not None, "If capacity is not given, must have path to load from"
                with open(path, "rb") as f:
                    data = np.load(f)
                    data = utils.nest_dict(data)
                self.add(data, data["label"])  # Add to the buffer via the add method!

        # Print the size of the allocation.
        storage = 0
        storage += 2 * get_buffer_bytes(self.obs_1_buffer)
        storage += 2 * get_buffer_bytes(self.action_1_buffer)
        storage += get_buffer_bytes(self.label_buffer)
        print("[PairwiseComparisonDataset] allocated {:.2f} GB".format(storage / 1024**3))

    def add(self, queries: Dict, labels: np.ndarray):
        assert self._capacity is not None, "Can only add to non-static buffers."
        assert (
            torch.utils.data.get_worker_info() is None
        ), "Cannot add to PairwiseComparisonDataset when parallelism is enabled."
        num_to_add = labels.shape[0]

        if self._idx + num_to_add > self._capacity:
            # We have more segments than capacity allows, complete in two writes.
            num_b4_wrap = self._capacity - self._idx
            self.add(utils.get_from_batch(queries, 0, num_b4_wrap), labels[:num_b4_wrap])
            self.add(utils.get_from_batch(queries, num_b4_wrap, num_to_add), labels[num_b4_wrap:])
        else:
            start, end = self._idx, self._idx + num_to_add
            utils.set_in_batch(self.obs_1_buffer, queries["obs_1"], start, end)
            utils.set_in_batch(self.obs_2_buffer, queries["obs_2"], start, end)
            utils.set_in_batch(self.action_1_buffer, queries["action_1"], start, end)
            utils.set_in_batch(self.action_2_buffer, queries["action_2"], start, end)
            self.label_buffer[start:end] = labels
            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def _sample(self, idxs):
        if self.subsample_size is None:
            obs_1 = utils.get_from_batch(self.obs_1_buffer, idxs)
            obs_2 = utils.get_from_batch(self.obs_2_buffer, idxs)
            action_1 = utils.get_from_batch(self.action_1_buffer, idxs)
            action_2 = utils.get_from_batch(self.action_2_buffer, idxs)
            label = self.label_buffer[idxs]
        else:
            # Note: subsample sequences currently do not support arbitrary obs/action spaces.
            start = np.random.randint(0, self.segment_size - self.subsample_size)
            end = start + self.subsample_size
            obs_1 = self.obs_1_buffer[idxs, start:end]
            obs_2 = self.obs_2_buffer[idxs, start:end]
            action_1 = self.action_1_buffer[idxs, start:end]
            action_2 = self.action_2_buffer[idxs, start:end]
            label = self.label_buffer[idxs]

        return dict(obs_1=obs_1, obs_2=obs_2, action_1=action_1, action_2=action_2, label=label, discount=self.discount)

    def save(self, path):
        # Save everything to the path via savez
        data = dict(
            obs_1=utils.get_from_batch(self.obs_1_buffer, 0, self._size),
            obs_2=utils.get_from_batch(self.obs_2_buffer, 0, self._size),
            action_1=utils.get_from_batch(self.action_1_buffer, 0, self._size),
            action_2=utils.get_from_batch(self.action_2_buffer, 0, self._size),
            label=self.label_buffer[: self._size],
        )
        data = utils.flatten_dict(data)
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **data)
            bs.seek(0)
            with open(path, "wb") as f:
                f.write(bs.read())

    def __len__(self):
        return self._size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        chunk_size = len(self) // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                cur_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                cur_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]
            yield self._sample(cur_idxs)
