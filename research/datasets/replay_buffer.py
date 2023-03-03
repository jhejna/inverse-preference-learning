import copy
import datetime
import io
import os
import random
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np
import torch

from research.utils import utils


def save_data(data: Dict, path: str) -> None:
    # Perform checks to make sure everything needed is in the data object
    assert all([k in data for k in ("obs", "action", "reward", "done", "discount")])
    # Flatten everything for saving as an np array
    data = utils.flatten_dict(data)
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
    data = utils.nest_dict(data)
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
            raise ValueError("Invalid value passed to `pad_ep`")


def get_buffer_bytes(buffer: np.ndarray) -> int:
    if isinstance(buffer, dict):
        return sum([get_buffer_bytes(v) for v in buffer.values()])
    elif isinstance(buffer, np.ndarray):
        return buffer.nbytes
    else:
        raise ValueError("Unsupported type passed to `get_buffer_bytes`.")


def remove_stack_dim(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: remove_stack_dim(v) for k, v in space.items()})
    elif isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(low=space.low[0], high=space.high[0])
    else:
        return space


class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    Generic Replay Buffer Class.

    This class adheres to the following conventions to support a wide array of multiprocessing options:
    1. Variables/functions starting with "_", like "_help" are to be used by the worker processes. This means
        they should be used only after __iter__ is called.
    2. variables/functions named regularly without a leading "_" are to be used by the main thread. This includes
        standard functions like "add".

    There are a few critical setup options.
    1. Distributed: this determines if the data is stored on the main processes, and then used via the shared address
        space. This will only work when multiprocessing is set to `fork` and not `spawn`.
        AKA it will duplicate memory on Windows and OSX!!!
    2. Capacity: determines if the buffer is setup upon creation. If it is set to a known value, then we can add data
        online with `add`, or by pulling more data from disk. If is set to None, the dataset is initialized to the full
        size of the offline dataset.
    3. batch_size: determines if we use a single sample or return entire batches

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
        distributed: bool = True,  # Whether or not the dataset is created in __init__ or __iter__. True means _-iter__
        path: Optional[str] = None,
        discount: float = 0.99,
        nstep: int = 1,
        cleanup: bool = True,
        fetch_every: int = 1000,  # How often to pull new data into the replay buffer.
        batch_size: Optional[int] = None,
        sample_multiplier: float = 1.5,  # Should be high enough so we always hit batch_size.
        stack: int = 1,
        pad: int = 0,
        next_obs: bool = True,  # Whether or not to load the next obs.
        stacked_obs: bool = False,  # Whether or not the data provided to the buffer will have stacked obs
        stacked_action: bool = False,  # Whether or not the data provided to the buffer will have stacked obs
    ):
        super().__init__()
        # Check that we don't over add in case of observation stacking
        self.stacked_obs = stacked_obs
        self.stacked_action = stacked_action
        if self.stacked_obs:
            observation_space = remove_stack_dim(observation_space)
        if self.stacked_action:
            action_space = remove_stack_dim(action_space)

        self.observation_space = observation_space
        self.action_space = action_space
        self.dummy_action = self.action_space.sample()

        # Data Storage parameters
        self.capacity = capacity  # The total storage of the dataset, or None if growth is disabled
        if self.capacity is not None:
            # Setup a storage path
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
            print("[research] Replay Buffer Storage Path", self.storage_path)
        self.distributed = distributed

        # Data Fetching parameters
        self.cleanup = cleanup
        self.path = path
        self.fetch_every = fetch_every
        self.sample_multiplier = sample_multiplier
        self.num_episodes = 0

        # Sampling values.
        self.discount = discount
        self.nstep = nstep
        self.stack = stack
        self.batch_size = 1 if batch_size is None else batch_size
        if pad > 0:
            assert self.stack > 1, "Pad > 0 doesn't make sense if we are not padding."
        self.pad = pad
        self.next_obs = next_obs

        if self.capacity is not None:
            # Print the total estimated data footprint used by the replay buffer.
            storage = 0
            storage += utils.np_bytes_per_instance(self.observation_space)
            storage += utils.np_bytes_per_instance(self.action_space)
            storage += utils.np_bytes_per_instance(0.0)  # Reward
            storage += utils.np_bytes_per_instance(0.0)  # Discount
            storage += utils.np_bytes_per_instance(False)  # Done
            storage = storage * capacity  # Total storage in Bytes.
            print("[ReplayBuffer] Estimated storage requirement for obs, action, reward, discount, done.")
            print("\t will not include kwarg storage: {:.2f} GB".format(storage / 1024**3))

        # Initialize in __init__ if the replay buffer is not distributed.
        if not self.distributed:
            print("[research] Replay Buffer not distributed. Alloc-ing in __init__")
            self._alloc()

    def _data_generator(self):
        """
        Can be overridden in order to load the initial data differently.
        By default assumes the data to be the standard format, and returned as:
        *(obs, action, reward, done, discount, kwargs)
        or
        None

        This function can be overriden by sub-classes in order to produce data batches.
        It should do the following:
        1. split data across torch data workers
        2. randomize the order of data
        3. yield data of the form (obs, action, reward, done, discount, kwargs)
        """
        if self.path is None:
            return
        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        ep_filenames = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".npz")]
        random.shuffle(ep_filenames)  # Shuffle all the filenames

        if num_workers > 1 and len(ep_filenames) == 1:
            print(
                "[ReplayBuffer] Warning: using multiple workers but single replay file. Reduce memory usage by sharding"
                " data with `save` instead of `save_flat`."
            )
        elif num_workers > 1 and len(ep_filenames) < num_workers:
            print("[ReplayBuffer] Warning: using more workers than dataset files.")

        for ep_filename in ep_filenames:
            ep_idx, _ = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            # Spread loaded data across workers if we have multiple workers and files.
            if ep_idx % num_workers != worker_id and len(ep_filenames) > 1:
                continue  # Only yield the files belonging to this worker.
            obs, action, reward, done, discount, kwargs = load_data(ep_filename)
            yield (obs, action, reward, done, discount, kwargs)

    def _alloc(self):
        """
        This function is responsible for allocating all of the data needed.
        It can be called in __init__ or during __iter___.

        It allocates all of the np buffers used to store data internal.
        It also sets the follow variables:
            _idx: internal _idx for the worker thread
            _size: internal _size of each workers dataset
            _current_data_generator: the offline data generator
            _loaded_all_offline_data: set to True if we don't need to load more offline data
        """
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        self._current_data_generator = self._data_generator()

        if self.capacity is not None:
            # If capacity was given, then directly alloc the buffers
            self._capacity = self.capacity // num_workers
            self._obs_buffer = utils.np_dataset_alloc(self.observation_space, self._capacity)
            self._action_buffer = utils.np_dataset_alloc(self.action_space, self._capacity)
            self._reward_buffer = utils.np_dataset_alloc(0.0, self._capacity)
            self._done_buffer = utils.np_dataset_alloc(False, self._capacity)
            self._discount_buffer = utils.np_dataset_alloc(0.0, self._capacity)
            self._kwarg_buffers = dict()
            self._size = 0
            self._idx = 0

            # Next, write in the alloced data lazily using the generator until we are full.
            preloaded_episodes = 0
            try:
                while self._size < self._capacity:
                    obs, action, reward, done, discount, kwargs = next(self._current_data_generator)
                    self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
                    preloaded_episodes += 1
                self._loaded_all_offline_data = False
            except StopIteration:
                self._loaded_all_offline_data = True  # We reached the end of the available dataset.

        else:
            self._capacity = None
            # Get all of the data and concatenate it together
            data = utils.concatenate(*list(self._current_data_generator), dim=0)
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
            self._loaded_all_offline_data = True

        # Print the size of the allocation.
        storage = 0
        storage += get_buffer_bytes(self._obs_buffer)
        storage += get_buffer_bytes(self._action_buffer)
        storage += get_buffer_bytes(self._reward_buffer)
        storage += get_buffer_bytes(self._done_buffer)
        storage += get_buffer_bytes(self._discount_buffer)
        storage += get_buffer_bytes(self._kwarg_buffers)
        print("[ReplayBuffer] Worker {:d} allocated {:.2f} GB".format(worker_id, storage / 1024**3))

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

        is_list = isinstance(reward, list) or isinstance(reward, np.ndarray)
        # Take only the last value if we are using stacking.
        # This prevents saving a bunch of extra data.
        if not is_list and self.stacked_obs:
            obs = utils.get_from_batch(obs, -1)
        if not is_list and action is not None and self.stacked_action:
            action = utils.get_from_batch(action, -1)

        if action is None:
            assert not isinstance(reward, (np.ndarray, list)), "Tried to add initial transition in batch mode."
            action = copy.deepcopy(self.dummy_action)
            reward = 0.0
            done = False
            discount = 1.0

        # Now we have multiple cases based on the transition type and parallelism of the dataset
        if not self.distributed:
            # We can add directly to the storage buffers.
            self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
            if self.cleanup:
                # If we are in cleanup mode, we don't keep the old data around. Immediately return
                return

        if not hasattr(self, "current_ep"):
            self.current_ep = dict()

        add_to_ep(self.current_ep, "obs", obs, is_list)
        add_to_ep(self.current_ep, "action", action, is_list)
        add_to_ep(self.current_ep, "reward", reward, is_list)
        add_to_ep(self.current_ep, "done", done, is_list)
        add_to_ep(self.current_ep, "discount", discount, is_list)
        add_to_ep(self.current_ep, "kwargs", kwargs, is_list)

        is_done = done[-1] if is_list else done
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
                utils.get_from_batch(obs, 0, num_b4_wrap),
                utils.get_from_batch(action, 0, num_b4_wrap),
                reward[:num_b4_wrap],
                done[:num_b4_wrap],
                discount[:num_b4_wrap],
                **utils.get_from_batch(kwargs, 0, num_b4_wrap),
            )
            self._add_to_buffer(
                utils.get_from_batch(obs, num_b4_wrap, num_to_add),
                utils.get_from_batch(action, num_b4_wrap, num_to_add),
                reward[num_b4_wrap:],
                done[num_b4_wrap:],
                discount[num_b4_wrap:],
                **utils.get_from_batch(kwargs, num_b4_wrap, num_to_add),
            )
        else:
            # Just add to the buffer
            start = self._idx
            end = self._idx + num_to_add
            utils.set_in_batch(self._obs_buffer, obs, start, end)
            utils.set_in_batch(self._action_buffer, action, start, end)
            utils.set_in_batch(self._reward_buffer, reward, start, end)
            utils.set_in_batch(self._done_buffer, done, start, end)
            utils.set_in_batch(self._discount_buffer, discount, start, end)

            for k, v in kwargs.items():
                if k not in self._kwarg_buffers:
                    sample_value = utils.get_from_batch(v, 0) if num_to_add > 1 else v
                    self._kwarg_buffers[k] = utils.np_dataset_alloc(sample_value, self._capacity)
                    print("[ReplayBuffer] Allocated", self._kwarg_buffers[k].bytes / 1024**3, "GB")
                utils.set_in_batch(self._kwarg_buffers[k], v, start, end)

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
            "obs": utils.get_from_batch(self._obs_buffer, 0, self._size),
            "action": utils.get_from_batch(self._action_buffer, 0, self._size),
            "reward": self._reward_buffer[: self._size],
            "done": self._done_buffer[: self._size],
            "discount": self._discount_buffer[: self._size],
            "kwargs": utils.get_from_batch(self._kwarg_buffers, 0, self._size),
        }
        os.makedirs(path, exist_ok=True)
        ep_len = len(data["reward"])
        ep_idx = 0
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
        save_path = os.path.join(path, ep_filename)
        save_data(data, save_path)
        return save_path

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

    def _fetch_online(self) -> None:
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

        # Return the fetched size
        return fetched_size

    def _fetch_offline(self) -> None:
        """
        This simple function fetches a new episode from the offline dataset and adds it to the buffer.
        This is done for each worker.
        """
        try:
            data = next(self._current_data_generator)
        except StopIteration:
            self._current_data_generator = self._data_generator()
            data = next(self._current_data_generator)

        obs, action, reward, done, discount, kwargs = data
        self._add_to_buffer(obs, action, reward, done, discount, **kwargs)
        # Return the fetched size
        return len(reward)

    def __iter__(self):
        assert not hasattr(self, "_iterated"), "__iter__ called twice!"
        self._iterated = True
        if self.distributed:
            # Allocate the buffer here if we are distributing across workers.
            self._alloc()

        # Setup variables for _fetch methods for getting new online data
        worker_info = torch.utils.data.get_worker_info()
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0
        assert self.distributed == (worker_info is not None), "ReplayBuffer.distributed set incorrectly."

        self._episode_filenames = set()
        self._samples_since_last_load = 0
        self._learning_online = False
        while True:
            yield self.sample(batch_size=self.batch_size, stack=self.stack, pad=self.pad)
            # Fetch new data...
            if self._capacity is not None:
                self._samples_since_last_load += 1
                if self._samples_since_last_load >= self.fetch_every:
                    # Fetch offline data
                    if not self._loaded_all_offline_data and not self._learning_online:
                        self._fetch_offline()
                    if self.distributed:  # If we are distributed we need to fetch the data.
                        fetch_size = self._fetch_online()
                        self._learning_online = fetch_size > 0
                    # Reset the fetch counter for this worker.
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

        obs = utils.get_from_batch(self._obs_buffer, obs_idxs)
        action = utils.get_from_batch(self._action_buffer, idxs)
        reward = np.zeros_like(self._reward_buffer[idxs])
        discount = np.ones_like(self._discount_buffer[idxs])
        for i in range(self.nstep):
            reward += discount * self._reward_buffer[idxs + i]
            discount *= self._discount_buffer[idxs + i] * self.discount

        kwargs = utils.get_from_batch(self._kwarg_buffers, next_obs_idxs)
        if self.next_obs:
            kwargs["next_obs"] = utils.get_from_batch(self._obs_buffer, next_obs_idxs)

        batch = dict(obs=obs, action=action, reward=reward, discount=discount, **kwargs)
        if pad > 0:
            batch["mask"] = self._compute_mask(idxs)

        return batch


class HindsightReplayBuffer(ReplayBuffer):
    """
    Modify the sample method of the ReplayBuffer to support hindsight sampling
    """

    def __init__(
        self,
        *args,
        reward_fn: Optional[Callable] = None,
        discount_fn: Optional[Callable] = None,
        goal_key: str = "desired_goal",
        achieved_key: str = "achieved_goal",
        strategy: str = "future",
        relabel_fraction: float = 0.5,
        mark_every: int = 100,
        init_obs: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.discount_fn = discount_fn
        self.goal_key = goal_key
        self.achieved_key = achieved_key
        self.strategy = strategy
        self.relabel_fraction = relabel_fraction
        self.mark_every = mark_every
        self.init_obs = init_obs
        assert isinstance(self.observation_space, gym.spaces.Dict), "HER Replay Buffer depends on Dict Spaces."

    def _extract_markers(self):
        # Write done at the idx position and the end of the buffer.
        idx_done, size_done = self._done_buffer[self._idx - 1], self._done_buffer[self._size - 1]
        self._done_buffer[self._idx - 1] = True  # mark true if index is before size
        self._done_buffer[self._size - 1] = True
        (self._ends,) = np.where(self._done_buffer[: self._size])
        self._starts = np.concatenate(([0], self._ends[:-1] + 1))
        # Clip the ends if they are at the same point
        if len(self._ends) > 0 and self._starts[-1] == self._ends[-1]:
            self._starts, self._ends = self._starts[:-1], self._ends[:-1]
        self._lengths = self._ends - self._starts + 1
        self._done_buffer[self._idx - 1], self._done_buffer[self._size - 1] = idx_done, size_done

    def _alloc(self):
        self._last_extract_size = 0
        super()._alloc()
        self._extract_markers()  # After we have allocated the dataset, extract markers
        self._last_extract_size = self._size

    def _add_to_buffer(self, obs: Any, action: Any, reward: Any, done: Any, discount: Any, **kwargs) -> None:
        super()._add_to_buffer(obs, action, reward, done, discount, **kwargs)
        if abs(self._idx - self._last_extract_size) >= self.mark_every:
            # Update the markers
            self._extract_markers()  # After we have allocated the dataset, extract markers

    def _get_one_idx(self, stack: int, pad: int) -> Union[int, np.ndarray]:
        # Sample an episode
        # Note: one problem with this approach is that we potentially over-sample
        # transitions from shorter episodes. In practice, however, this decision is
        # often made for speed.
        ep_idx = np.random.randint(0, len(self._starts))
        # Sample an idx in the dataset
        sample_limit = self._lengths[ep_idx] - self.nstep * (self.stack - pad)
        if sample_limit <= 0:
            return self._get_one_idx(stack, pad)
        pos = np.random.randint(0, sample_limit) + 1  # Can't sample the first
        idx = self._starts[ep_idx] + pos
        if stack > 1:
            idx = idx + np.arange(stack) * self.nstep
        return ep_idx, idx

    def _get_many_idxs(self, batch_size: int, stack: int, pad: int, depth: int = 0) -> np.ndarray:
        ep_idxs = np.random.randint(0, len(self._starts), size=int(self.sample_multiplier * batch_size))
        sample_limit = self._lengths[ep_idxs] - self.nstep * (stack - pad)
        valid_idxs = sample_limit > 0
        ep_idxs = ep_idxs[valid_idxs]
        sample_limit = sample_limit[valid_idxs]
        if len(ep_idxs) < batch_size and depth < 20:
            print(
                "[research ReplayBuffer] Buffer Sampler did not recieve batch_size number of valid indices. Consider"
                " increasing sample_multiplier."
            )
            return self._get_many_idxs(batch_size, stack, pad, depth=depth + 1)
        # Get all the valid samples
        ep_idxs = ep_idxs[:batch_size]
        sample_limit = sample_limit[:batch_size]
        pos = np.random.randint(0, sample_limit) + 1
        idxs = self._starts[ep_idxs] + pos
        if stack > 1:
            stack_idxs = np.arange(stack) * self.nstep
            idxs = np.expand_dims(idxs, axis=-1) + stack_idxs
        return ep_idxs, idxs

    def sample(self, batch_size: Optional[int] = None, stack: int = 1, pad: int = 0):
        if len(self._ends) == 0 or self._size <= self.nstep * stack + 2:  # Must have at least one completed episode
            return {}

        if batch_size > 1:
            ep_idxs, idxs = self._get_many_idxs(batch_size, stack, pad)
        else:
            ep_idxs, idxs = self._get_one_idx(stack, pad)
            # If we are only sampling once expand dims to match the multi case
            idxs = np.expand_dims(idxs, axis=0)
            ep_idxs = np.expand_dims(ep_idxs, axis=0)

        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1
        last_idxs = next_obs_idxs[..., -1] if stack > 1 else next_obs_idxs

        obs = utils.get_from_batch(self._obs_buffer, obs_idxs)
        action = utils.get_from_batch(self._action_buffer, idxs)
        kwargs = utils.get_from_batch(self._kwarg_buffers, next_obs_idxs)

        if "horizon" in kwargs:
            horizon = kwargs["horizon"]
        else:
            horizon = -100 * np.ones_like(idxs, dtype=np.int)

        her_idxs = np.where(np.random.uniform(size=idxs.shape) < self.relabel_fraction)

        if self.strategy == "last":
            goal_idxs = self._ends[ep_idxs[her_idxs]]
        elif self.strategy == "next":
            # It was whatever was achieved in the next obs
            goal_idxs = np.minimum(self._ends[ep_idxs[her_idxs]], last_idxs[her_idxs])
        else:
            # add 1 to go the the end of the episode
            goal_idxs = np.random.randint(last_idxs[her_idxs], self._ends[ep_idxs[her_idxs]] + 1)

        # Compute the horizon
        if self.nstep > 1:
            horizon[her_idxs] = np.ceil((goal_idxs - obs_idxs[her_idxs]) / self.nstep).astype(np.int)
        else:
            horizon[her_idxs] = goal_idxs - obs_idxs[her_idxs]

        # If we use the any strategy, sample some random goals
        if self.strategy.startswith("any"):
            # Relabel part of the goals to anything, other goals are relabeled with future
            # The percentage sampled marignally is indicated by the fraction. ie any_0.5 will sample
            # 50% of the goals randomly from the achieved states.
            parts = self.strategy.split("_")
            if len(parts) == 1:
                parts.append(1)
            if self.batch_size == 1:
                any_split = 1 if np.random.random() < float(parts[1]) else 0
            else:
                any_split = int(her_idxs[0].shape[0] * float(parts[1]))
            goal_idxs[:any_split] = np.random.randint(0, self._size, size=any_split)  # sample any index
            # Adjust horizon
            any_idxs = (her_idxs[0][:any_split],)
            horizon[any_idxs] = -100  # set them back to -100, the default mask value

        if len(idxs.shape) == 2:
            goal_idxs = np.expand_dims(goal_idxs, axis=-1)  # Add the stack dimension

        # Relabel
        if self.relabel_fraction < 1.0:
            desired = obs[self.goal_key].copy()
            desired[her_idxs] = self._obs_buffer[self.achieved_key][goal_idxs]
        else:
            desired = self._obs_buffer[self.achieved_key][goal_idxs]
        kwargs["horizon"] = horizon

        reward = np.zeros_like(idxs, dtype=np.float32)
        discount = np.ones_like(idxs, dtype=np.float32)
        for i in range(self.nstep):
            achieved = self._obs_buffer[self.achieved_key][idxs + i]
            reward += discount * self.reward_fn(achieved, desired)
            step_discount = self.discount_fn(achieved, desired) if self.discount_fn is not None else 1.0
            discount *= step_discount * self.discount

        # Write observations
        obs[self.goal_key] = desired
        if self.next_obs:
            next_obs = utils.get_from_batch(self._obs_buffer, next_obs_idxs)
            next_obs[self.goal_key] = desired
            kwargs["next_obs"] = next_obs
        if self.init_obs:
            init_idxs = self._starts[ep_idxs]
            if stack > 1:
                init_idxs = np.expand_dims(init_idxs, axis=-1) + np.arange(stack) * self.nstep
            init_obs = utils.get_from_batch(self._obs_buffer, init_idxs)
            init_obs[self.goal_key] = desired
            kwargs["init_obs"] = init_obs
        if pad > 0:
            kwargs["mask"] = self._compute_mask(idxs)

        # TODO: support relabeling reward-to-go.
        assert "rtg" not in kwargs

        batch = dict(obs=obs, action=action, reward=reward, discount=discount, **kwargs)
        if batch_size == 1:
            batch = utils.squeeze(batch, 0)
        return batch
