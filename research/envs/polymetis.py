import os
import time

import gym
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface


class PolyMetisEnv(gym.Env):
    def __init__(
        self,
        ip_address="localhost",
        port=50051,
        max_delta=0.05,
        initialization_noise=0.0,
        time_to_go=0.1,
        fix_gripper=False,
        use_quat=False,
        ee_link=7,
    ):
        super().__init__()
        # Connect to the robot. This could be a simulator or the actual panda
        self.robot = RobotInterface(ip_address=ip_address, port=port)
        self.fix_gripper = fix_gripper
        if self.fix_gripper:
            self.action_space = gym.spaces.Box(low=-1, high=1.0, shape=(3,))
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1.0, shape=(4,))
            self.gripper = GripperInterface()

        self.ee_link = ee_link
        if self.ee_link == 7:
            self._desired_quat = torch.tensor([0.923879564, -0.382683456, 0.0, 0.0])
        elif self.ee_link == 11:
            self._desired_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Invalid EE Link provided")

        # Values taken from:
        # https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/conf/robot_client/
        self.ee_pos_lower_limit = np.array([0.1, -0.45, 0.11])
        self.ee_pos_upper_limit = np.array([1.0, 0.45, 1.0])
        self.max_detla = max_delta
        self.initialization_noise = initialization_noise
        self.time_to_go = time_to_go
        self.use_quat = use_quat
        self.robot.start_cartesian_impedance()

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action):
        # First, clip the action to the desired range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # remove the gripper action
        if not self.fix_gripper:
            gripper_action = action[3]
        action = action[:3]
        action = action * self.max_detla
        # Clip the action so we don't leave the area
        action = torch.clamp(
            self._ee_pos + action, min=torch.tensor(self.ee_pos_lower_limit), max=torch.tensor(self.ee_pos_upper_limit)
        )
        self.robot.update_desired_ee_pose(position=action, orientation=self._desired_quat)  # self._desired_quat)

        if not self.fix_gripper:
            current_gripper_state = self.gripper.get_state().width
            open_or_close = 1 if gripper_action < 0 else -1
            speed = 0.01
            gripper_action = current_gripper_state + open_or_close * speed
            gripper_action = np.clip(gripper_action, -1, 1)
            self.gripper.goto(width=gripper_action, speed=2 * speed, force=0.1, blocking=False)

        time.sleep(self.time_to_go)
        self._ee_pos, self._ee_quat = self.robot.get_ee_pose()

        state = self._compute_state().copy()
        reward = self._compute_reward()
        done = self._compute_done()
        return state, reward, done, {}

    def reset(self):
        self.robot.terminate_current_policy()
        self.robot.go_home()
        self._ee_pos, self._ee_quat = self.robot.get_ee_pose()
        self._desired_quat = self._ee_quat

        if self.initialization_noise > 0:
            delta = self.initialization_noise * np.random.uniform(low=-1, high=1, size=self._ee_pos.shape)
            ee_pos = self._ee_pos.cpu().numpy() + delta
            time_to_go = 20 * self.time_to_go if self.time_to_go is not None else None
            self.robot.move_to_ee_pose(position=ee_pos, orientation=self._desired_quat, time_to_go=time_to_go)

        self.robot.start_cartesian_impedance()
        self._ee_pos, self._ee_quat = self.robot.get_ee_pose()
        return self._compute_state().copy()

    def _get_obs(self):
        raise NotImplementedError

    def _get_reward(self):
        raise NotImplementedError

    def _get_done(self):
        raise NotImplementedError


class PolyMetisReach(PolyMetisEnv):
    def __init__(self, *args, goal=(0.8, 0.2, 0.1), state_noise=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = np.array(goal)

    @property
    def observation_space(self):
        n_dims = 3
        if self.use_quat:
            n_dims += 4
        if not self.fix_gripper:
            n_dims += 2

        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_dims,))

    def _compute_state(self):
        obs_list = [self._ee_pos.cpu().numpy()]
        if self.use_quat:
            obs_list.append(self._ee_quat.cpu().numpy())
        if not self.fix_gripper:
            width = self.gripper.get_state().width
            obs_list.append(np.array([-width / 2, width / 2]))
        return np.concatenate(obs_list, axis=0)

    def _compute_reward(self):
        pos = self._ee_pos.numpy()
        assert self.goal.shape == pos.shape
        return -0.1 * np.linalg.norm(pos - self.goal)

    def _compute_done(self):
        pos = self._ee_pos.numpy()
        assert self.goal.shape == pos.shape
        dist = np.linalg.norm(pos - self.goal)  # tolerance
        print(dist)
        done = dist < 0.01
        if done:
            print("SUCCESS!")
        return done

    def update_goal(self, goal):
        self.goal = np.array(goal)


class CubeInterface:
    def __init__(self, path="/tmp/cube_estimator.txt"):
        self.path = path
        self.fd = open(self.path, "rb")
        self._last_state = None

    def update(self):
        try:
            self.fd.seek(-2, os.SEEK_END)
            while self.fd.read(1) != b"\n":
                self.fd.seek(-2, os.SEEK_CUR)
        except OSError:
            self.fd.seek(0)
        last_line = self.fd.readline().decode()
        self._last_state = np.fromstring(last_line.strip()[1:-1], dtype=float, sep=" ")

    def get_state(self):
        self.update()
        return self._last_state


class PolyMetisBlockPush(PolyMetisEnv):
    def __init__(self, *args, goal=[0.35, 0.35], state_noise=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = np.array(goal)
        self.cube = CubeInterface()

    @property
    def observation_space(self):
        n_dims = 5  # Add block obs
        if self.use_quat:
            n_dims += 4
        if not self.fix_gripper:
            n_dims += 2
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_dims,))

    def _compute_state(self):
        obs_list = [self._ee_pos.cpu().numpy()]
        if self.use_quat:
            obs_list.append(self._ee_quat.cpu().numpy())
        if not self.fix_gripper:
            width = self.gripper.get_state().width
            obs_list.append(np.array([-width / 2, width / 2]))
        # Append the block observation
        obs_list.append(self.cube.get_state()[:2])
        print(obs_list)
        return np.concatenate(obs_list, axis=0)

    def _compute_reward(self):
        return 0

    def _compute_done(self):
        dist = np.linalg.norm(self.cube.get_state()[:2] - self.goal)
        print(dist)
        done = dist < 0.05
        if done:
            print("SUCCESS")
        return done

    def update_goal(self, goal):
        self.goal = np.array(goal)
