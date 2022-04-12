from json import load
from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import tempfile
import io
import gym
import collections
import copy
import datetime
import os
import shutil

def save_episode(episode, path):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(path, 'wb') as f:
            f.write(bs.read())

def load_episode(path):
    with open(path, 'rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBuffer(torch.utils.data.IterableDataset):
    '''
    This replay buffer is carefully implemented to run efficiently and prevent multiprocessing
    memory leaks and errors.
    All variables starting with an underscore ie _variable are used only by the child processes
    All other variables are used by the parent process.
    '''
    def __init__(self, observation_space, action_space, 
                       discount=0.99, batch_size=None):
        # Observation and action space values
        self.observation_space = observation_space
        self.action_space = action_space

        # Queuing values
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size

    @property
    def is_full(self):
        return False
    
    def __del__(self):
        pass

    def add(self):
        pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Rollout Buffer does not support worker parallelism at the moment."
        pass
