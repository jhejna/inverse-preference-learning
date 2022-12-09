import copy
import datetime
import io
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch

from research.utils.utils import concatenate, flatten_dict, get_from_batch, nest_dict, np_dataset_alloc, set_in_batch


def save_data(data: Dict, path: str) -> None:
    # Perform checks to make sure everything needed is in the data object
    assert all([k in data for k in ("obs", "action", "reward", "done", "discount")])
    # Flatten everything for saving as an np array
    data = flatten_dict(data)
    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if not isinstance(data[k], np.ndarray):
            assert isinstance(data[k], list), "Unknown type passed to save_data"
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, int):
                dtype = np.int64
            elif isinstance(first_value, bool):
                dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)

    length = len(data["reward"])
    assert all([len(data[k]) == length for k in data.keys()])

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def load_data(path: str) -> Dict:
    with open(path, "rb") as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    # Unnest the data to get everything in the correct format
    data = nest_dict(data)
    kwargs = data.get("kwargs", dict())
    return data["obs"], data["action"], data["reward"], data["done"], data["discount"], kwargs


def add_to_ep(d: Dict, key: str, value: Any, extend: bool = False) -> None:
    # I don't really like this function because it modifies d in place.
    # Perhaps later this can be refactored.
    # If this key isn't the dict, we need to append it
    if key not in d:
        if isinstance(value, dict):
            d[key] = dict()
        else:
            d[key] = list()
    # If the value is a dict, then we need to traverse to the next level.
    if isinstance(value, dict):
        for k, v in value.items():
            add_to_ep(d[key], k, v, extend=extend)
    else:
        if extend:
            d[key].extend(value)
        else:
            d[key].append(value)


def add_dummy_transition(d: Dict, length: int):
    # Helper method to add the dummy transition if it wasn't already
    for k in d.keys():
        if isinstance(d[k], dict):
            add_dummy_transition(d[k], length)
        elif isinstance(d[k], list):
            assert len(d[k]) == length or len(d[k]) == length - 1
            if len(d[k]) == length - 1:
                d[k].insert(0, d[k][0])  # Duplicate the first item.
        else:
            raise ValueError("Invalid value passed to pad_ep")


class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    Generic Replay Buffer Class
    This class adheres to the following conventions to support a wide array of multiprocessing options:
    1. Variables/functions starting with "_", like "_help" are to be used by the worker processes. This means
        they should be used only after __iter__ is called.
    2. variables/functions named regularly are to be used by the main thread

    There are a few critical setup options.
    1. Distributed: this determines if the data is stored on the main processes, and then used via the shared address
        space. This will only work when multiprocessing is set to fork and not spawn.
        AKA it will duplicate memory on Windows and OSX
    2. Capacity: determines if the buffer is setup for Online RL (can add data), or offline (cannot add data)
    3. batch_size: determines if we use a single sample or return entire batches
    4. dataloader workers: determines how we setup the sharing.

    Some options are mutually exclusive. For example, it is bad to use a non-distributed layout with
    workers and online data. This will generate a bunch of copy on writes.

    Data is expected to be stored in a "next" format. This means that data is stored like this:
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1
    s_3, a_2  , r_2  , d_2 ... End of episode!
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1

    This format is expected from the load(path) funciton.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: Optional[int] = None,
        distributed: bool = True,
        path: Optional[str] = None,
        discount: float = 0.99,
        nstep: int = 1,
        cleanup: bool = True,
        fetch_every: int = 1000,
        batch_size: Optional[int] = None,
        sample_multiplier: float = 1.5,
        stack: int = 1,
        pad: int = 0,
        next_obs: bool = True,
    ):
        # run initial argument checks
        self.observation_space = observation_space
        self.action_space = action_space
        self.dummy_action = self.action_space.sample()

        # Data Storage parameters
        self.capacity = capacity  # The total storage of the dataset, or None if growth is disabled
        self.distributed = distributed  # Whether or not the dataset is created in __init__ or __iter__
        self.cleanup = cleanup
        self.path = path
        self.fetch_every = fetch_every
        self.sample_multiplier = sample_multiplier
        self.num_episodes = 0

        # Sampling values
        self.discount = discount
        self.nstep = nstep
        self.stack = stack
        self.batch_size = 1 if batch_size is None else batch_size
        if pad > 0:
            assert self.stack > 1, "Pad > 0 doesn't make sense if we are not padding."
        self.pad = pad
        self.next_obs = next_obs

        if not self.distributed:
            print("[research] Replay Buffer not distributed. Alloc-ing in __init__")
            self._alloc()
        elif self.capacity is not None:
            # Setup a storage path
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
            print("[research] Replay Buffer Storage Path", self.storage_path)

    def preload(self):
        """
        Can be overridden in order to load the initial data differently.
        By default assumes the data to be the standard format, and returned as:
        *(obs, action, reward, done, discount, kwargs)
        or
        None
        """
        if self.path is None:
            return
        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        ep_filenames = sorted([os.path.join(self.path, f) for f in os.listdir(self.path)], reverse=True)
        for ep_filename in ep_filenames:
            ep_idx, _ = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_idx % num_workers != worker_id:
                continue
            # try:
            obs, action, reward, done, discount, kwargs = load_data(ep_filename)
            yield (obs, action, reward, done, discount, kwargs)

            # except:
            #     continue

    def _alloc(self):
        """
        This function is responsible for allocating all of the data needed.
        It can be called in __init__ or during __iter___
        """

        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        if self.capacity is not None:
            # If capacity was given, then directly alloc the buffers
            self._capacity = self.capacity // num_workers
            self._obs_buffer = np_dataset_alloc(self.observation_space, self._capacity)
            self._action_buffer = np_dataset_alloc(self.action_space, self._capacity)
            self._reward_buffer = np_dataset_alloc(0.0, self._capacity)
            self._done_buffer = np_dataset_alloc(False, self._capacity)
            self._discount_buffer = np_dataset_alloc(0.0, self._capacity)
            self._kwarg_buffers = dict()
            self._size = 0
            self._idx = 0

            # Next, write in the alloced data lazily using the generator.
            for data in self.preload():
                obs, action, reward, done, discount, kwargs = data
                self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
        else:
            self._capacity = None
            # Get all of the data and concatenate it together
            data = concatenate(*list(self.preload()), dim=0)
            obs, action, reward, done, discount, kwargs = data
            self._obs_buffer = obs
            self._action_buffer = action
            self._reward_buffer = reward
            self._done_buffer = done
            self._discount_buffer = discount
            self._kwarg_buffers = kwargs
            # Set the size to be the shape of the reward buffer
            self._size = self._reward_buffer.shape[0]
            self._idx = self._size
            assert self._size > 0, "Fixed replay buffer found no data at specified path."

    def add(
        self,
        obs: Any,
        action: Optional[Union[Dict, np.ndarray]] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        discount: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Make sure that if we are adding the first transition, it is consistent
        assert self.capacity is not None, "Tried to extend to a static size buffer."
        assert (action is None) == (reward is None) == (done is None) == (discount is None)
        if action is None:
            assert not isinstance(reward, (np.ndarray, list)), "Tried to add initial transition in batch mode."
            action = copy.deepcopy(self.dummy_action)
            reward = 0.0
            done = False
            discount = 1.0

        # Now we have multiple cases based on the transition type and parallelism of the dataset
        if not self.is_parallel:
            # We can add directly
            self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
            if self.cleanup:
                # If we are in cleanup mode, we don't keep the old data around. Immediately return
                return

        if not hasattr(self, "current_ep"):
            self.current_ep = dict()

        is_list = isinstance(reward, list) or isinstance(reward, np.ndarray)
        is_done = done[-1] if is_list else done

        add_to_ep(self.current_ep, "obs", obs, is_list)
        add_to_ep(self.current_ep, "action", action, is_list)
        add_to_ep(self.current_ep, "reward", reward, is_list)
        add_to_ep(self.current_ep, "done", done, is_list)
        add_to_ep(self.current_ep, "discount", discount, is_list)
        add_to_ep(self.current_ep, "kwargs", kwargs, is_list)

        if is_done:
            # Dump the data
            ep_idx = self.num_episodes
            ep_len = len(self.current_ep["reward"])
            # Check to make sure that kwargs are the same length
            add_dummy_transition(self.current_ep, ep_len)
            self.num_episodes += 1
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
            save_data(self.current_ep, os.path.join(self.storage_path, ep_filename))
            self.current_ep = dict()

    def _add_to_buffer(self, obs: Any, action: Any, reward: Any, done: Any, discount: Any, **kwargs) -> None:
        # Can add in batches or serially.
        if isinstance(reward, list) or isinstance(reward, np.ndarray):
            num_to_add = len(reward)
            assert num_to_add > 1, "If inputting lists or arrays should have more than one timestep"
        else:
            num_to_add = 1

        if self._idx + num_to_add > self._capacity:
            # Add all we can at first, then add the rest later
            num_b4_wrap = self._capacity - self._idx
            self._add_to_buffer(
                get_from_batch(obs, 0, num_b4_wrap),
                get_from_batch(action, 0, num_b4_wrap),
                reward[:num_b4_wrap],
                done[:num_b4_wrap],
                discount[:num_b4_wrap],
                **get_from_batch(kwargs, 0, num_b4_wrap),
            )
            self._add_to_buffer(
                get_from_batch(obs, num_b4_wrap, num_to_add),
                get_from_batch(action, num_b4_wrap, num_to_add),
                reward[num_b4_wrap:],
                done[num_b4_wrap:],
                discount[num_b4_wrap:],
                **get_from_batch(kwargs, num_b4_wrap, num_to_add),
            )
        else:
            # Just add to the buffer
            start = self._idx
            end = self._idx + num_to_add
            set_in_batch(self._obs_buffer, obs, start, end)
            set_in_batch(self._action_buffer, action, start, end)
            set_in_batch(self._reward_buffer, reward, start, end)
            set_in_batch(self._done_buffer, done, start, end)
            set_in_batch(self._discount_buffer, discount, start, end)

            for k, v in kwargs.items():
                if k not in self._kwarg_buffers:
                    sample_value = get_from_batch(v, 0) if num_to_add > 1 else v
                    self._kwarg_buffers[k] = np_dataset_alloc(sample_value, self._capacity)
                set_in_batch(self._kwarg_buffers[k], v, start, end)

            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def save(self, path: str) -> None:
        """
        Save the replay buffer to the specified path. This is literally just copying the files
        from the storage path to the desired path. By default, we will also delete the original files.
        """
        if self.cleanup:
            print("[research] Warning, attempting to save a cleaned up replay buffer. There are likely no files")
        os.makedirs(path, exist_ok=True)
        srcs = os.listdir(self.storage_path)
        for src in srcs:
            shutil.move(os.path.join(self.storage_path, src), os.path.join(path, src))
        print("Successfully saved", len(srcs), "episodes.")

    def save_flat(self, path):
        """
        Save directly from the buffers instead of from the saved data. This saves everything as a flat file.
        """
        assert self._size != 0, "Trying to flat save a buffer with no data."
        data = {
            "obs": get_from_batch(self._obs_buffer, 0, self._size),
            "action": get_from_batch(self._action_buffer, 0, self._size),
            "reward": self._reward_buffer[: self._size],
            "done": self._done_buffer[: self._size],
            "discount": self._discount_buffer[: self._size],
            "kwargs": get_from_batch(self._kwarg_buffers, 0, self._size),
        }
        os.makedirs(path, exist_ok=True)
        ep_len = len(data["reward"])
        ep_idx = 0
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
        save_data(data, os.path.join(path, ep_filename))

    def __del__(self):
        if not self.cleanup:
            return
        if hasattr(self, "storage_path"):
            paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
            for path in paths:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                os.rmdir(self.storage_path)
            except:
                pass

    def _fetch(self) -> None:
        """
        Fetches data from the storage path
        """
        ep_filenames = sorted([os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_idx % self._num_workers != self._worker_id:
                continue
            if ep_filename in self._episode_filenames:
                break  # We found something we have already loaded
            if fetched_size + ep_len > self._capacity:
                break  # Cannot fetch more than the size of the replay buffer
            # Load the episode from disk
            obs, action, reward, done, discount, kwargs = load_data(ep_filename)
            self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
            self._episode_filenames.add(ep_filename)
            if self.cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

    @property
    def is_parallel(self) -> bool:
        return not hasattr(self, "_is_serial")

    def setup(self):
        """
        This function will "setup" the replay buffer.
        After calling this we cannot use the buffer in parallel.
        """
        self._is_serial = True
        if self.distributed:
            self._alloc()

    def __iter__(self):
        assert not hasattr(self, "_iterated"), "__iter__ called twice!"
        self._iterated = True
        worker_info = torch.utils.data.get_worker_info()
        if hasattr(self, "_is_serial") and self._is_serial:
            # The buffer has already been alloced.
            assert worker_info is None, "Cannot use parallel after setting up"
        else:
            # If we have not already been setup, alloc the buffer if distributed.
            if self.distributed:
                self._alloc()

        # Now setup variables for _fetch
        if worker_info is None:
            self._is_serial = True
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0
        self._episode_filenames = set()
        self._samples_since_last_load = 0

        while True:
            yield self.sample(batch_size=self.batch_size, stack=self.stack, pad=self.pad)
            if self.capacity is not None and self.is_parallel:
                self._samples_since_last_load += 1
                if self._samples_since_last_load >= self.fetch_every:
                    self._fetch()
                    self._samples_since_last_load = 0

    def _get_one_idx(self, stack: int, pad: int) -> Union[int, np.ndarray]:
        # Add 1 for the first dummy transition
        idx = np.random.randint(0, self._size - self.nstep * stack) + 1
        done_idxs = idx + np.arange(self.nstep * (stack - pad)) - 1
        if np.any(self._done_buffer[done_idxs]):
            # If the episode is done at any point in the range, we need to sample again!
            # Note that we removed the pad length, as we can check the padding later
            return self._get_one_idx(stack, pad)
        if stack > 1:
            idx = idx + np.arange(stack) * self.nstep
        return idx

    def _get_many_idxs(self, batch_size: int, stack: int, pad: int, depth: int = 0) -> np.ndarray:
        idxs = np.random.randint(0, self._size - self.nstep * stack, size=int(self.sample_multiplier * batch_size)) + 1

        done_idxs = np.expand_dims(idxs, axis=-1) + np.arange(self.nstep * (stack - pad)) - 1
        valid = np.logical_not(
            np.any(self._done_buffer[done_idxs], axis=-1)
        )  # Compute along the done axis, not the index axis.

        valid_idxs = idxs[valid == True]  # grab only the idxs that are still valid.
        if len(valid_idxs) < batch_size and depth < 20:  # try a max of 20 times
            print(
                "[research ReplayBuffer] Buffer Sampler did not recieve batch_size number of valid indices. Consider"
                " increasing sample_multiplier."
            )
            return self._get_many_idxs(batch_size, stack, pad, depth=depth + 1)
        idxs = valid_idxs[:batch_size]  # Return the first [:batch_size] of them.
        if stack > 1:
            stack_idxs = np.arange(stack) * self.nstep
            idxs = np.expand_dims(idxs, axis=-1) + stack_idxs
        return idxs

    def _compute_mask(self, idxs: np.ndarray) -> np.ndarray:
        # Check the validity via the done buffer to determine the padding mask
        mask = np.zeros(idxs.shape, dtype=np.bool_)
        for i in range(self.nstep):
            mask = (
                mask + self._done_buffer[idxs + (i - 1)]
            )  # Subtract one when checking for parity with index sampling.
        # Now set everything past the first true to be true
        mask = np.minimum(np.cumsum(mask, axis=-1), 1.0)
        return mask

    def sample(self, batch_size: Optional[int] = None, stack: int = 1, pad: int = 0) -> Dict:
        if self._size <= self.nstep * stack + 2:
            return {}
        # NOTE: one small bug is that we won't end up being able to sample segments that span
        # Across the barrier of the replay buffer. We lose 1 to self.nstep transitions.
        # This is only a problem if we keep the capacity too low.
        if batch_size > 1:
            idxs = self._get_many_idxs(batch_size, stack, pad)
        else:
            idxs = self._get_one_idx(stack, pad)
        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1

        obs = get_from_batch(self._obs_buffer, obs_idxs)
        action = get_from_batch(self._action_buffer, idxs)
        reward = np.zeros_like(self._reward_buffer[idxs])
        discount = np.ones_like(self._discount_buffer[idxs])
        for i in range(self.nstep):
            reward += discount * self._reward_buffer[idxs + i]
            discount *= self._discount_buffer[idxs + i] * self.discount

        kwargs = get_from_batch(self._kwarg_buffers, next_obs_idxs)
        if self.next_obs:
            kwargs["next_obs"] = get_from_batch(self._obs_buffer, next_obs_idxs)

        batch = dict(obs=obs, action=action, reward=reward, discount=discount, **kwargs)
        if pad > 0:
            batch["mask"] = self._compute_mask(idxs)

        return batch
