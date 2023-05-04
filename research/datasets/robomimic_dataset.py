from typing import Optional

import gym
import h5py
import numpy as np
import torch

from research.utils.utils import remove_float64

from .replay_buffer import ReplayBuffer


class RobomimicDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the RoboMimicDatasets into a ReplayBuffer

    While I don't agree with all the decisions made here, this was designed to exactly
    match the behavior in the PreferenceTransformer Codebase when using the defaults.

    See https://github.com/csmile-1006/PreferenceTransformer/JaxPref/reward_transform.py#L437
    """

    def __init__(self, *args, eps: Optional[float] = 1e-5, separate_validation=False, **kwargs):
        self.eps = eps
        self.separate_validation = separate_validation
        super().__init__(*args, **kwargs)

    def _data_generator(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Goal Conditioned Robomimic Dataset does not support sharded loading."
        # Get the keys to use
        keys = [
            "object",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]

        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        f = h5py.File(self.path, "r")

        if self.separate_validation:
            demos = [
                elem.decode("utf-8") for elem in np.array(f["mask/train"][:])
            ]  # Extract the training demonstrations
        else:
            demos = list(f["data"].keys())  # Otherwise use all demos

        for i, demo in enumerate(demos):
            if i % num_workers != worker_id and len(demos) > num_workers:
                continue
            obs = np.concatenate(
                [f["data"][demo]["obs"][k] for k in keys],
                axis=1,
            )
            final_obs = np.concatenate(
                [f["data"][demo]["next_obs"][k][-1:] for k in keys],
                axis=1,
            )
            obs = np.concatenate((obs, final_obs), axis=0)  # Concat on time axis
            obs = remove_float64(obs)

            dummy_action = np.expand_dims(self.dummy_action, axis=0)
            action = np.concatenate((dummy_action, f["data"][demo]["actions"]), axis=0)
            action = remove_float64(action)

            if self.eps is not None:
                lim = 1 - self.eps
                action = np.clip(action, -lim, lim)

            reward = np.concatenate(([0], f["data"][demo]["rewards"]), axis=0)
            reward = remove_float64(reward)

            done = np.concatenate(([0], f["data"][demo]["dones"]), axis=0).astype(np.bool_)
            done[-1] = True

            discount = (1 - done).astype(np.float32)
            assert len(obs) == len(action) == len(reward) == len(done) == len(discount)
            kwargs = dict()
            yield (obs, action, reward, done, discount, kwargs)

        f.close()  # Close the file handler.
