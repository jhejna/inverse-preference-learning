import gym
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client


class PandaEnv(gym.Env):
    def __init__(
        self,
        basePosition=[0, 0, 0],
        max_delta=0.05,
        fix_gripper=False,
        initialization_noise=0.0,
        use_quat=False,
        n_substeps=20,
        timestep=0.002,
        ee_link=7,
    ):
        # Setup the simulator
        self.urdfRootPath = pybullet_data.getDataPath()
        self.sim = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.sim.setTimeStep(timestep)
        self.sim.resetSimulation()
        self.sim.setGravity(0, 0, -9.81)
        # Setup the camera
        self._camera_width, self._camera_height = None, None
        self.sim.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=30, cameraPitch=-60, cameraTargetPosition=[0.5, -0.2, 0.0]
        )
        self.view_matrix = self.sim.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0], distance=1.0, yaw=90, pitch=-50, roll=0, upAxisIndex=2
        )
        self._projection_matrix = None
        # setup the panda
        self.sim.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        self.panda = self.sim.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=basePosition)

        # Save parameters
        self.initialization_noise = initialization_noise
        self.max_delta = max_delta
        self.fix_gripper = fix_gripper
        self.use_quat = use_quat
        self.ee_pos_lower_limit = np.array([0.1, -0.4, -0.05])
        self.ee_pos_upper_limit = np.array([1.0, 0.4, 1.0])
        self.ee_link = ee_link
        if ee_link == 7:
            self._desired_quat = np.array([0.923879564, -0.382683456, 0.0, 0.0])
        elif ee_link == 11:
            self._desired_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Invalid EE Link provided")
        self.max_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
        self.n_substeps = n_substeps

        # Set the joint limits as in polymetis
        if self.fix_gripper:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action):
        ee_state, ee_quat = self._ee_state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_action = action[:3]
        ee_desired = ee_state + self.max_delta * ee_action

        # Update the robot
        new_joint_positions = self.sim.calculateInverseKinematics(
            self.panda, self.ee_link, list(ee_desired), self._desired_quat
        )
        self.sim.setJointMotorControlArray(
            self.panda,
            range(9),
            self.sim.POSITION_CONTROL,
            targetPositions=list(new_joint_positions),
            forces=self.max_forces,
        )

        # Update the gripper
        if not self.fix_gripper:
            gripper_action = action[3]
            speed = 0.025
            open_or_close = 1 if gripper_action < 0 else -1
            new_width = self._gripper_width + open_or_close * speed
            new_width = min(new_width, 0.1)
            gripper_position = [new_width / 2, new_width / 2]
        else:
            gripper_position = [0.0, 0.0]
        self.sim.setJointMotorControlArray(
            self.panda, [9, 10], self.sim.POSITION_CONTROL, targetPositions=list(gripper_position)
        )

        for _ in range(self.n_substeps):
            self.sim.stepSimulation()

        success = self._get_success()
        info = dict() if success is None else dict(success=success)
        return self._get_obs(), self._get_reward(), self._get_done(), info

    def reset(self):
        init_pos = [0.0, 0.0, 0.0, -2 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0, 0.0]
        if self.fix_gripper:
            init_pos += [0.0, 0.0]
        else:
            init_pos += [0.05, 0.05]
        for idx in range(len(init_pos)):
            self.sim.resetJointState(self.panda, idx, init_pos[idx])
        ee_pos, ee_quat = self._ee_state
        self._desired_quat = ee_quat

        if self.initialization_noise > 0:
            ee_pos = np.array(ee_pos)
            delta = self.initialization_noise * np.random.uniform(low=-1, high=1, size=ee_pos.shape)
            ee_desired = np.clip(ee_pos + delta, self.ee_pos_lower_limit, self.ee_pos_upper_limit)
            # Update the robot
            new_joint_positions = self.sim.calculateInverseKinematics(
                self.panda, self.ee_link, list(ee_desired), list(ee_quat)
            )
            for i, pos in enumerate(new_joint_positions):
                self.sim.resetJointState(self.panda, i, pos)

        self._reset()
        self.sim.stepSimulation()
        return self._get_obs()

    def render(self, mode="rgb_array", width=120, height=120):
        if height != self._camera_height or width != self._camera_width or self._projection_matrix is None:
            self._camera_width, self._camera_height = width, height
            self._projection_matrix = self.sim.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
        (width, height, pxl, _, _) = self.sim.getCameraImage(
            width=width, height=height, viewMatrix=self.view_matrix, projectionMatrix=self._projection_matrix
        )
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    @property
    def _ee_state(self):
        """
        return pos, quat of link 7. This is the base of the hand, and doesn't include the hand itself! This is critical
        """
        ee_state = self.sim.getLinkState(self.panda, self.ee_link)
        return np.array(ee_state[4]), np.array(ee_state[5])

    @property
    def _grasp_state(self):
        grasp_state = self.sim.getLinkState(self.panda, 11)
        return np.array(grasp_state[4]), np.array(grasp_state[5])

    @property
    def _gripper_width(self):
        joint_pos = np.array(self.sim.getJointState(self.panda, 9)[0])
        return 2 * joint_pos

    @property
    def _gripper_state(self):
        width = self._gripper_width
        return np.array([-width / 2, width / 2])

    @property
    def _joint_position(self):
        joint_states = self.sim.getJointStates(self.panda, range(9))
        return np.array([joint_state[0] for joint_state in joint_states])

    def _reset(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _get_reward(self):
        raise NotImplementedError

    def _get_done(self):
        raise NotImplementedError

    def _get_success(self):
        return None


class Reach(PandaEnv):
    def __init__(self, *args, goal=[0.5602202, 0.35655773, 0.11], **kwargs):
        self.goal = np.array(goal)
        assert len(goal) == 3
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self):
        n_dims = 3
        if self.use_quat:
            n_dims += 4
        if not self.fix_gripper:
            n_dims += 2
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_dims,))

    def _get_obs(self):
        ee_pos, ee_quat = self._ee_state
        obs_list = [ee_pos]
        if self.use_quat:
            obs_list.append(ee_quat)
        if not self.fix_gripper:
            obs_list.append(self._gripper_state)
        obs = np.concatenate(obs_list, axis=0).copy()
        return obs

    def _get_reward(self):
        ee_pos, _ = self._ee_state
        dist = np.linalg.norm(ee_pos - self.goal)
        reward = -0.1 * dist
        return reward

    def _get_success(self):
        ee_pos, _ = self._ee_state
        dist = np.linalg.norm(ee_pos - self.goal)
        return dist < 0.02

    def _get_done(self):
        return False


class BlockPush(PandaEnv):
    def __init__(
        self,
        *args,
        goal=(0.35, 0.35),
        randomize_block=True,
        block_obs=True,
        dist_reward=0.02,
        grasp_bonus=0.0,
        state_noise=0.0,
        **kwargs,
    ):
        self.goal = np.array(goal)
        assert len(goal) == 2
        super().__init__(*args, **kwargs)
        self.randomize_block = randomize_block
        self.block_obs = block_obs
        self.block_size = 0.05
        self.block_mass = 2.0  # Set block mass to 1.5.
        self.block_base_position = np.array((0.5, 0, 0.05 / 2))
        self.dist_reward = dist_reward
        self.grasp_bonus = grasp_bonus
        self.state_noise = state_noise

        baseVisualShapeIndex = self.sim.createVisualShape(
            self.sim.GEOM_BOX,
            halfExtents=np.ones(3) * self.block_size / 2,
            specularColor=np.zeros(3),
            rgbaColor=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        baseCollisionShapeIndex = self.sim.createCollisionShape(
            self.sim.GEOM_BOX, halfExtents=np.ones(3) * self.block_size / 2
        )
        self.block = self.sim.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=self.block_mass,
            basePosition=self.block_base_position + np.array([0.0, 0.0, 0.01]),  # add a Z offest for initial placement
        )
        self.sim.changeDynamics(bodyUniqueId=self.block, linkIndex=-1, lateralFriction=0.5)

    @property
    def observation_space(self):
        n_dims = 3  # EE Pos and X, Y position of block
        if self.use_quat:
            n_dims += 4
        if not self.fix_gripper:
            n_dims += 2
        if self.block_obs:
            n_dims += 2
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_dims,))

    def _get_obs(self):
        ee_pos, ee_quat = self._ee_state
        obs_list = [ee_pos]
        if self.use_quat:
            obs_list.append(ee_quat)
        if not self.fix_gripper:
            obs_list.append(self._gripper_state)
        if self.block_obs:
            obs_list.append(self._block_pos[:2])
        obs = np.concatenate(obs_list, axis=0).copy()
        obs += self.state_noise * np.clip(np.random.randn(*obs.shape), -1.5, 1.5)
        return obs

    def _get_reward(self):
        # Get the vector between the two end effector points
        grasp_point = self._grasp_state[0]
        dist_to_block = np.linalg.norm(grasp_point - self._block_pos)
        margin = self.block_size / 2 + 0.0075
        dist_to_block = max(dist_to_block, margin) - margin

        goal_pos = np.array([self.goal[0], self.goal[1], self.block_size / 2])
        dist_to_target = np.linalg.norm(goal_pos - self._block_pos)

        reward = -self.dist_reward * dist_to_block + -0.1 * dist_to_target

        if self.grasp_bonus > 0:
            if self.is_grasping():
                reward += self.grasp_bonus

        return reward

    def _get_success(self):
        dist_to_target = np.linalg.norm(self.goal - self._block_pos[:2])
        return dist_to_target < 0.03

    def _get_done(self):
        return False

    def _reset(self):
        if self.randomize_block:
            noise_limits = np.array([0.15, 0.15, 0.0])
            noise = np.random.uniform(low=-noise_limits, high=noise_limits)
        else:
            noise = np.zeros(3)
        pos = self.block_base_position + noise
        pos += np.array([0.0, 0.0, 0.01])
        orn = (0, 0, 0, 1)
        self.sim.resetBasePositionAndOrientation(self.block, pos, orn)

    def is_grasping(self):
        # check if there is any contact on the internal part of the fingers, to control if they
        # are correctly touching an object
        obj_id = self.block
        idx_fingers = [9, 10]

        p0 = self.sim.getContactPoints(obj_id, self.panda, linkIndexB=idx_fingers[0])
        p1 = self.sim.getContactPoints(obj_id, self.panda, linkIndexB=idx_fingers[1])

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = self.sim.getLinkState(self.panda, idx_fingers[0])[4:6]
            f0_pos_w = self.sim.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = self.sim.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = self.sim.getLinkState(self.panda, idx_fingers[1])[4:6]
            f1_pos_w = self.sim.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = self.sim.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p0_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)

    @property
    def _block_pos(self):
        return np.array(self.sim.getBasePositionAndOrientation(self.block)[0])


if __name__ == "__main__":
    env = BlockPush(use_quat=False, fix_gripper=False, initialization_noise=0.0, randomize_block=True, grasp_bonus=0.0)
    obs = env.reset()
    done = False
    max_length = 100
    while not done and max_length > 0:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        max_length -= 1
        import time

        time.sleep(0.1)
    print("Finished")
