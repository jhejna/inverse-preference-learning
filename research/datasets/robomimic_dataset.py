import gym
import h5py
import numpy as np
import torch

from research.utils.utils import remove_float64

from .replay_buffer import ReplayBuffer


class RobomimicDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the RoboMimicDatasets into a ReplayBuffer
    """

    def __init__(self, *args, **kwargs):
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
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/train"][:])]  # Extract the training demonstrations

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

            reward = np.concatenate(([0], f["data"][demo]["rewards"]), axis=0)
            reward = remove_float64(reward)

            done = np.zeros(action.shape[0], dtype=np.bool_)  # Gets recomputed with HER
            done[-1] = True
            discount = np.ones(action.shape[0])  # Gets recomputed with HER
            assert len(obs[self.achieved_key]) == len(action) == len(reward) == len(done) == len(discount)
            kwargs = dict()
            yield (obs, action, reward, done, discount, kwargs)

        f.close()  # Close the file handler.
