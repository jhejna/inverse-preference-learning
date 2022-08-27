"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gym
import metaworld
import numpy as np


class SawyerEnv(gym.Env):
    def __init__(self, env_name, seed=True):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = self._env.max_path_length
        self.max_episode_steps = self._env.max_path_length

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Add the underlying state to the info
        state = self._env.sim.get_state()
        info["state"] = np.concatenate((state.qpos, state.qvel), axis=0)
        return obs.astype(np.float32), reward, done, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        self._env.set_state(qpos, qvel)

    def reset(self, **kwargs):
        self._episode_steps = 0
        return self._env.reset(**kwargs).astype(np.float32)

    def render(self, mode="rgb_array", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        view_1 = self._env.render(offscreen=True, camera_name="corner", resolution=(width, height))
        view_2 = self._env.render(offscreen=True, camera_name="topview", resolution=(width, height))
        return np.concatenate((view_1, view_2), axis=0)
