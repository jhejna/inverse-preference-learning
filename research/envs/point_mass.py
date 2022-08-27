# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("point_mass.xml"), common.ASSETS


@SUITE.add()
def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None, target_pos=(-1, 1), target_size=0.015, environment_kwargs=None):
    """Returns the hard point_mass task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PointMassGoal(target_pos=target_pos, target_size=target_size, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def reach_l2(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, target_pos=(-1, 1), target_size=0.015, environment_kwargs=None
):
    """Returns the hard point_mass task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PointMassGoalL2(target_pos=target_pos, target_size=target_size, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def reach_l2_random(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, target_pos=(-1, 1), target_size=0.015, environment_kwargs=None
):
    """Returns the hard point_mass task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PointMassGoalL2Random(target_pos=target_pos, target_size=target_size, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""

    def mass_to_target(self):
        """Returns the vector from mass to target in global coordinate."""
        return self.named.data.geom_xpos["target"] - self.named.data.geom_xpos["pointmass"]

    def mass_to_target_dist(self):
        """Returns the distance from mass to the target."""
        return np.linalg.norm(self.mass_to_target())


class PointMassGoal(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, target_pos=(1, 0), target_size=0.015, random=None):
        self._target_size = target_size
        x, y = target_pos
        self._target_pos = (0.27 * x, 0.27 * y)
        super().__init__(random=random)

    def initialize_episode(self, physics):
        # Setup the the episode with the specified goal location and size
        # Note that the goal is specified in terms of -1, 1 ranges.
        physics.named.model.geom_size["target", 0] = self._target_size
        physics.named.model.geom_pos["target", "x"] = self._target_pos[0]
        physics.named.model.geom_pos["target", "y"] = self._target_pos[1]
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_size = physics.named.model.geom_size["target", 0]
        near_target = rewards.tolerance(physics.mass_to_target_dist(), bounds=(0, target_size), margin=target_size)
        control_reward = rewards.tolerance(physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (control_reward + 4) / 5
        return near_target * small_control


class PointMassGoalL2(PointMassGoal):
    """A point_mass `Task` to reach target with L2 Reward and early termination"""

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_reward = -1 * physics.mass_to_target_dist()
        control_reward = -1 * np.square(physics.control()).mean()
        return target_reward + 0.1 * control_reward

    def get_termination(self, physics):
        target_size = physics.named.model.geom_size["target", 0]
        dist = physics.mass_to_target_dist()
        if dist <= target_size:
            return 0.0  # Return discount factor of zero as the MDP terminates.
        else:
            return None  # Episode is not done!


class PointMassGoalL2Random(PointMassGoal):
    """A point_mass `Task` to reach target with L2 Reward and early termination"""

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_reward = -1 * physics.mass_to_target_dist()
        control_reward = -1 * np.square(physics.control()).mean()
        return target_reward + 0.1 * control_reward

    def initialize_episode(self, physics):
        # This randomizes the initial position of the point mass
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # Setup the the episode with the specified goal location and size
        # Note that the goal is specified in terms of -1, 1 ranges.
        physics.named.model.geom_size["target", 0] = self._target_size
        physics.named.model.geom_pos["target", "x"] = self._target_pos[0]
        physics.named.model.geom_pos["target", "y"] = self._target_pos[1]
        super().initialize_episode(physics)

    def get_termination(self, physics):
        target_size = physics.named.model.geom_size["target", 0]
        dist = physics.mass_to_target_dist()
        if dist <= target_size:
            return 0.0  # Return discount factor of zero as the MDP terminates.
        else:
            return None  # Episode is not done!
