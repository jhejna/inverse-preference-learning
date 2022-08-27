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

"""Reacher domain."""

import collections

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 10  # Reduce time limit to 10 from 20. Episode is 500 steps
_SMALL_TARGET = 0.015


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("reacher.xml"), common.ASSETS


@SUITE.add()
def reach_l2(time_limit=_DEFAULT_TIME_LIMIT, target_pos=(0, 1), random=None, environment_kwargs=None):
    """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = ReacherGoal(target_pos=target_pos, target_size=_SMALL_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""

    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return self.named.data.geom_xpos["target", :2] - self.named.data.geom_xpos["finger", :2]

    def finger_to_target_dist(self):
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class ReacherGoal(base.Task):
    """A reacher `Task` to reach the target."""

    def __init__(self, target_pos=(0, 1), target_size=_SMALL_TARGET, random=None):
        """Initialize an instance of `Reacher`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target_size = target_size
        angle, radius = target_pos
        assert 0 <= angle <= 2 * np.pi, "Invalid target position given"
        radius *= 0.2
        assert 0.05 <= radius <= 0.2, "Invalid radius specified"
        self._target_pos = (angle, radius)
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.named.model.geom_size["target", 0] = self._target_size
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # Randomize target position
        angle, radius = self._target_pos
        physics.named.model.geom_pos["target", "x"] = radius * np.sin(angle)
        physics.named.model.geom_pos["target", "y"] = radius * np.cos(angle)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """
        Returns an observation of the state.
        NOTE: removed the inclusion of the finger to target to make the task harder!
        """
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_reward = -1 * physics.finger_to_target_dist()
        control_reward = -1 * np.square(physics.control()).mean()
        return target_reward + 0.1 * control_reward

    def get_termination(self, physics):
        # Edit this to no discount factor, thus the final position has to be stable
        return None
        # this is new
        # if physics.finger_to_target_dist() <= 0.5*self._target_size: # halve the size
        #     return 0.0 # Return discount factor of zero as the MDP terminates.
        # else:
        #     return None # Episode is not done!
