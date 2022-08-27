import collections
import datetime
import io
import math
import os
import random
import shutil
import tempfile

import numpy as np
import torch

from research.utils.utils import get_from_batch

from .replay_buffer import construct_buffer_helper, load_episode, save_episode


class FeedbackLabelDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        path,
        discount=0.99,
        segment_size=20,
        nstep=1,
        batch_size=64,
        replay_capacity=100000,
        segment_capacity=100000,
        cleanup=False,
        preload_path=None,
    ):
        super().__init__()
        # Setup the replay buffer data
        self.storage_path = path
        self.discount = discount
        self.replay_capacity = replay_capacity
        self._size = 0
        self._episode_filenames_and_lengths = []
        self._episodes = {}
        self.cleanup = cleanup

        # Parameters for segments
        self.segment_size = segment_size
        self.nstep = nstep
        self.batch_size = batch_size
        self.segment_capacity = segment_capacity

        # Run mandatory checks on the datatypes
        # Unfortunately, this currently only supports Box spaces
        import gym

        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Box)
        # Construct the buffers
        self.obs_1_buffer = construct_buffer_helper(
            observation_space, self.segment_capacity, begin_pad=(self.segment_size,)
        )
        self.obs_2_buffer = construct_buffer_helper(
            observation_space, self.segment_capacity, begin_pad=(self.segment_size,)
        )
        self.action_1_buffer = construct_buffer_helper(
            action_space, self.segment_capacity, begin_pad=(self.segment_size,)
        )
        self.action_2_buffer = construct_buffer_helper(
            action_space, self.segment_capacity, begin_pad=(self.segment_size,)
        )
        self.label_buffer = construct_buffer_helper(0.5, self.segment_capacity)
        # init buffer parameters
        self.buffer_index = 0
        self.buffer_size = 0
        self._recent_queries = None

        if preload_path is not None:
            self._load(preload_path)  # Collect data from the other dataset.

    def _load(self, path):
        # Reload the set of valid data from the replay buffer
        ep_filenames = sorted([os.path.join(path, f) for f in os.listdir(path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_filename in self._episodes:
                break  # We found something we have already loaded
            if fetched_size + ep_len > self.replay_capacity:
                break  # Cannot fetch more than the size of the replay buffer
            # Add the length to the fetched size
            fetched_size += ep_len
            while ep_len + self._size > self.replay_capacity:
                # Delete episodes from the list
                early_ep_filename, early_ep_len = self._episode_filenames_and_lengths.pop(0)
                self._size -= early_ep_len
                if early_ep_filename in self._episodes:  # Remove from storage.
                    del self._episodes[early_ep_filename]
                # To allow for sufficient delete time, cleanup episodes on unlink here.
                if self.cleanup:
                    try:
                        os.remove(early_ep_filename)
                    except OSError:
                        pass
            self._episode_filenames_and_lengths.append((ep_filename, ep_len))  # Add the new episode and it's length
            self._episode_filenames_and_lengths.sort(key=lambda x: x[0])

    def _get_episode(self, ep_filename):
        if ep_filename not in self._episodes:
            # Try to load the episode from disk
            try:
                episode = load_episode(ep_filename)
            except:
                return None
            self._episodes[ep_filename] = episode
        return self._episodes.get(ep_filename, None)

    def _get_segment(self):
        episode = None
        ep_len = 0
        while (
            episode is None or ep_len - self.segment_size * self.nstep < 1
        ):  # Ensure that the episode is also long enough!
            ep_name, ep_len = random.choice(self._episode_filenames_and_lengths)
            episode = self._get_episode(ep_name)

        # Get a random start index
        idx = np.random.choice(ep_len - self.segment_size * self.nstep)
        action_segment = (
            {k: v[idx : idx + self.segment_size * self.nstep] for k, v in episode["action"].items()}
            if isinstance(episode["action"], dict)
            else episode["action"][idx : idx + self.segment_size * self.nstep]
        )
        obs_segment = (
            {k: v[idx : idx + self.segment_size * self.nstep] for k, v in episode["obs"].items()}
            if isinstance(episode["obs"], dict)
            else episode["obs"][idx : idx + self.segment_size * self.nstep]
        )
        if self.nstep > 1:
            action_segment = (
                {k: v[:: self.nstep] for k, v in action_segment.items()}
                if isinstance(action_segment, dict)
                else action_segment[:: self.nstep]
            )
            obs_segment = (
                {k: v[:: self.nstep] for k, v in obs_segment.items()}
                if isinstance(obs_segment, dict)
                else obs_segment[:: self.nstep]
            )
        # Run np sum prod
        reward = episode["reward"][idx : idx + self.segment_size * self.nstep]  # Compute discounted reward
        reward = np.sum(reward * np.power(self.discount, np.arange(reward.shape[0])))

        # Get the state if it exists
        if "kwargs_state" in episode:
            state_segment = episode["kwargs_state"][idx : idx + self.segment_size * self.nstep][:: self.nstep]
        else:
            state_segment = None
        return obs_segment, action_segment, reward, state_segment

    def get_segments(self, batch_size=64):
        self._load(path=self.storage_path)  # Reload data
        if len(self._episode_filenames_and_lengths) == 0:
            return None
        obs_list, action_list, reward_list, state_list = [], [], [], []
        for _ in range(2 * batch_size):
            obs, action, reward, state = self._get_segment()
            # add to temporary buffer
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            state_list.append(state)
        obs = np.stack(obs_list, axis=0)
        action = np.stack(action_list, axis=0)
        reward = np.array(reward_list)
        if state_list[0] is None:
            state = obs
        else:
            state = np.stack(state_list, axis=0)

        return dict(
            obs_1=obs[:batch_size],
            action_1=action[:batch_size],
            reward_1=reward[:batch_size],
            state_1=state[:batch_size],
            obs_2=obs[batch_size:],
            action_2=action[batch_size:],
            reward_2=reward[batch_size:],
            state_2=state[batch_size:],
        )

    def label_segments(self, batch, label):
        batch = {k: v[: len(label)] for k, v in batch.items()}
        self._recent_queries = {k: v for k, v in batch.items()}
        self._recent_queries["label"] = label  # The most recent batch should also include state information
        # Modify the batch to remove the queries with -1 labels
        valid_idxs = label != -1
        num_valid_labels = np.sum(valid_idxs)
        if num_valid_labels == 0:
            return
        elif num_valid_labels < len(valid_idxs):
            label = label[valid_idxs]
            batch = {k: v[valid_idxs] for k, v in batch.items()}

        def add_helper(x, y):
            num_segments = len(y)
            self.obs_1_buffer[self.buffer_index : self.buffer_index + num_segments] = x["obs_1"]
            self.obs_2_buffer[self.buffer_index : self.buffer_index + num_segments] = x["obs_2"]
            self.action_1_buffer[self.buffer_index : self.buffer_index + num_segments] = x["action_1"]
            self.action_2_buffer[self.buffer_index : self.buffer_index + num_segments] = x["action_2"]
            self.label_buffer[self.buffer_index : self.buffer_index + num_segments] = y
            self.buffer_index = (self.buffer_index + num_segments) % self.segment_capacity
            self.buffer_size = min(self.segment_capacity, self.buffer_size + num_segments)

        if self.buffer_index + len(label) > self.segment_capacity:
            # We can't do a regular write, we need to split it up.
            num_before_wrap = self.segment_capacity - self.buffer_index
            # First do the write for the part that fits
            add_helper({k: v[:num_before_wrap] for k, v in batch.items()}, label[:num_before_wrap])
            # Then do the after wrap
            add_helper({k: v[num_before_wrap:] for k, v in batch.items()}, label[num_before_wrap:])
        else:
            add_helper(batch, label)

    def _sample(self, idxs):
        obs_1 = self.obs_1_buffer[idxs]
        obs_2 = self.obs_2_buffer[idxs]
        action_1 = self.action_1_buffer[idxs]
        action_2 = self.action_2_buffer[idxs]
        label = self.label_buffer[idxs]
        # Note: we don't need to sample the states for this
        return dict(obs_1=obs_1, obs_2=obs_2, action_1=action_1, action_2=action_2, label=label)

    def __len__(self):
        return self.buffer_size

    def get_queries_for_visualization(self, num=4):
        # return the most recent queries, yielding just the state information
        return {k: v[:num] for k, v in self._recent_queries.items()}

    def __iter__(self):
        assert torch.utils.data.get_worker_info() is None, "FeedbackLabel Dataset is not designed for parallelism."
        idxs = np.random.permutation(len(self))
        for i in range(math.ceil(len(self) / self.batch_size)):  # Need to use ceil to get all data points.
            cur_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]
            yield self._sample(cur_idxs)

    # Add functionality for saving the buffer
    def save(self, path):
        buffer = {
            "obs_1": self.obs_1_buffer[: self.buffer_size],
            "obs_2": self.obs_2_buffer[: self.buffer_size],
            "action_1": self.action_1_buffer[: self.buffer_size],
            "action_2": self.action_2_buffer[: self.buffer_size],
            "label": self.label_buffer[: self.buffer_size],
        }
        save_episode(buffer, path)


class MultiTaskOracleFeedbackDataset(torch.utils.data.IterableDataset):
    def __init__(self, observation_space, action_space, paths, segment_capacity, **kwargs):
        super().__init__()
        if isinstance(paths, str):
            paths = [os.path.join(paths, p) for p in os.listdir(paths)]

        # Get the basenames of the directories to use as task names
        task_names = [os.path.basename(p) for p in paths]
        self._task_to_id = {task: index for index, task in enumerate(task_names)}
        self._id_to_task = {v: k for k, v in self._task_to_id.items()}
        self._datasets = {}
        for task, p in zip(task_names, paths):
            dataset = FeedbackLabelDataset(observation_space, action_space, p, **kwargs)
            # Determine if we are dealing with a replay buffer or a predefined feedback array.
            # We do this by looking to see if we have saved feedback in the folder
            if "feedback.npz" in os.listdir(p):
                batch = load_episode(os.path.join(p, "feedback.npz"))
                end_index = len(batch["label"])
                start_index = max(0, end_index - segment_capacity)
                batch = get_from_batch(batch, start=start_index, end=end_index)
                dataset.label_segments(batch, batch["label"])
            else:
                # Otherwise, generate the queries.
                batch = dataset.get_segments(batch_size=segment_capacity)
                label = 1.0 * (batch["reward_1"] < batch["reward_2"])
                dataset.label_segments(batch, label)
                # Now remove all the loaded episodes to save memory
                dataset._episode_filenames_and_lengths = {}
                ep_names = list(dataset._episodes.keys())
                for name in ep_names:
                    del dataset._episodes[name]
                dataset._episodes = {}

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


class CrossPolicyOracleFeedbackDataset(torch.utils.data.IterableDataset):
    def __init__(self, observation_space, action_space, paths, segment_capacity, **kwargs):
        super().__init__()
        if isinstance(paths, str):
            paths = [os.path.join(paths, p) for p in os.listdir(paths)]

        # Get the basenames of the directories to use as task names
        task_names = [os.path.basename(p) for p in paths]
        self._task_to_id = {task: index for index, task in enumerate(task_names)}
        self._id_to_task = {v: k for k, v in self._task_to_id.items()}
        self._datasets = {
            task: FeedbackLabelDataset(observation_space, action_space, p, segment_capacity=segment_capacity, **kwargs)
            for task, p in zip(task_names, paths)
        }

        # Initialize all of the datasets by generating queries and labeling them.
        dataset_keys = list(self._datasets.keys())
        for task, dataset in self._datasets.items():
            # Generate the queries once a time, going across different datasets.
            # Do this once at a time
            for _ in range(segment_capacity):
                positive_segment = dataset.get_segments(batch_size=1)
                negative_segment = self._datasets[random.choice(dataset_keys)].get_segments(batch_size=1)
                if random.random() > 0.5:
                    label = 1
                    batch = dict(
                        obs_1=negative_segment["obs_1"],
                        action_1=negative_segment["action_1"],
                        obs_2=positive_segment["obs_2"],
                        action_2=positive_segment["action_2"],
                    )
                else:
                    label = 0
                    batch = dict(
                        obs_1=positive_segment["obs_1"],
                        action_1=positive_segment["action_1"],
                        obs_2=negative_segment["obs_2"],
                        action_2=negative_segment["action_2"],
                    )
                dataset.label_segments(batch, label)
            # Now remove all the loaded episodes to save memory
            dataset._episodes = {}
            print("Finished", task)

    def __iter__(self):
        assert torch.utils.data.get_worker_info() is None, "FeedbackLabel Dataset is not designed for parallelism."
        iterators = {task: iter(dataset) for task, dataset in self._datasets.items()}
        # The dict order should be shuffled.
        while True:
            tasks = list(iterators.keys())
            random.shuffle(tasks)
            for task in tasks:
                iterator = iterators[task]
                try:
                    batch = next(iterator)
                except StopIteration:
                    # We ran out of batches in one of the datasets. This is approximately one epoch, so return.
                    return
                yield (self._task_to_id[task], batch)
