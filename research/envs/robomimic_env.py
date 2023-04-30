import gym
import h5py
import numpy as np
from robomimic.config import config_factory
from robomimic.utils import env_utils, file_utils, obs_utils
from robosuite.wrappers import GymWrapper


class RoboMimicEnv(gym.Env):
    def __init__(self, path, horizon=500):
        env_meta = file_utils.get_env_metadata_from_dataset(dataset_path=path)
        env_meta["env_kwargs"]["horizon"] = horizon
        env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        ).env
        env.ignore_done = False
        env._max_episode_steps = env.horizon

        self.env = GymWrapper(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        if self.env._check_success():
            done = True

        return observation, reward, done, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
