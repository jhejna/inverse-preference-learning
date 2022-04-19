import torch
import numpy as np
import math

from .replay_buffer import construct_buffer_helper

class RolloutBuffer(torch.utils.data.IterableDataset):
    '''

    '''
    def __init__(self, observation_space, action_space, 
                       discount=0.99, batch_size=None,
                       gae_lambda=0.95, capacity=2048):
        # Observation and action space values
        self.observation_space = observation_space
        self.action_space = action_space

        # Queuing values
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.batch_size = 1 if batch_size is None else batch_size
        self._capacity = capacity + 2 # Add one for the first timestep and one for the last timestep
        self._last_batch = True
        self._idx = 0

    @property
    def is_full(self):
        return self._idx >= self._capacity

    @property
    def last_batch(self):
        return self._last_batch

    def setup(self):
        # Setup the required rollout buffers
        self._obs_buffer = construct_buffer_helper(self.observation_space, self._capacity)
        self._action_buffer = construct_buffer_helper(self.action_space, self._capacity)
        self._reward_buffer = construct_buffer_helper(0.0, self._capacity)
        self._done_buffer = construct_buffer_helper(False, self._capacity)
        self._info_buffers = dict()
        self._idx = 0

    def __del__(self):
        pass

    def add(self, obs, action=None, reward=None, done=None, **kwargs):
        assert (action is None) == (reward is None) == (done is None)
        if action is None:
            # TODO: figure out if we should have the discount factor here.
            action = self.action_space.sample()
            reward = 0.0
            done = False

        assert self._idx < self._capacity, "Called add on a full buffer"
        def add_to_buffer_helper(buffer, value):
            if isinstance(buffer, dict):
                for k, v in buffer.items():
                    add_to_buffer_helper(v, value[k])
            elif isinstance(buffer, np.ndarray):
                buffer[self._idx] = value
            else:
                raise ValueError("Attempted buffer ran out of space!")

        add_to_buffer_helper(self._obs_buffer, obs.copy())
        add_to_buffer_helper(self._action_buffer, action.copy())
        add_to_buffer_helper(self._reward_buffer, reward)
        add_to_buffer_helper(self._done_buffer, done)

        for k, v in kwargs.items():
            if k not in self._info_buffers:
                self._info_buffers[k] = construct_buffer_helper(v, self._capacity)
            add_to_buffer_helper(self._info_buffers[k], v.copy())

        self._idx += 1 # increase the index

    def prepare_buffer(self):
        assert "value" in self._info_buffers, "Attempted to use Rollout Buffer but values were not added."
        self._advantage_buffer = construct_buffer_helper(0.0, self._capacity)
        
        last_gae_lam = 0
        for step in reversed(range(1, self._capacity - 1)): # Stay within the valid range
            next_non_terminal = 1.0 - self._done_buffer[step] # Get done from the current step. Maybe should be step + 1? But i think not.
            next_values = self._info_buffers["value"][step + 1]

            delta = self._reward_buffer[step] + self.discount * next_values * next_non_terminal - self._info_buffers["value"][step]
            last_gae_lam = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae_lam
            self._advantage_buffer[step] = last_gae_lam
        
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self._return_buffer = self._advantage_buffer + self._info_buffers["value"]

    def _get(self, idxs):
        if idxs.shape[0] == 1:
            idxs = idxs[0]

        obs_idxs = idxs - 1
        obs = {k:v[obs_idxs] for k, v in self._obs_buffer} if isinstance(self._obs_buffer, dict) else self._obs_buffer[obs_idxs]
        action = {k:v[idxs] for k, v in self._action_buffer} if isinstance(self._action_buffer, dict) else self._action_buffer[idxs]
        returns = self._return_buffer[idxs]
        advantage = self._advantage_buffer[idxs]
        
        batch = dict(obs=obs, action=action, returns=returns, advantage=advantage)
        for k, v in self._info_buffers.items():
            batch[k] = v[idxs]
        return batch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Rollout Buffer does not support worker parallelism at the moment."
        # Return Empty Batches if we are not full
        if not self.is_full:
            self._last_batch = True
            yield dict()
            return
        
        self.prepare_buffer()
        self._last_batch = False
        idxs = np.random.permutation(self._capacity - 2) + 1 # Add one offset for initial observation
        num_batches = math.ceil(len(idxs) / self.batch_size)
        for i in range(num_batches - 1): # Do up to the last 
            batch_idxs = idxs[i*self.batch_size:(i+1)*self.batch_size]
            yield self._get(batch_idxs)
        self._last_batch = True # Flag last batch
        last_idxs = idxs[(num_batches - 1)*self.batch_size:]
        yield self._get(last_idxs)
