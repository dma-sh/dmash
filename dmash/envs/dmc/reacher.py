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

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015


def update_physics(xml_string, length=1.0, action_factor=1.0):
    """
    Adapts and returns the xml_string of the model with the given context.
    Inspired by
    https://github.com/automl/CARL/blob/main/carl/envs/dmc/dmc_tasks/utils.py
    """

    mjcf = etree.fromstring(xml_string)

    # Update length and mass
    bodies = mjcf.findall(".//body")
    for body in bodies:
        if body.get("name") in ["hand", "finger"]:
            pos = body.get("pos").split(" ")
            pos = " ".join([str(float(pos[0]) * length), pos[1], pos[2]])
            body.set("pos", pos)
    geoms = mjcf.findall(".//geom")
    for geom in geoms:
        if geom.get("name") in ["arm", "hand"]:
            fromto = geom.get("fromto").split(" ")
            fromto = " ".join([fromto[0], fromto[1], fromto[2], str(float(fromto[3]) * length), fromto[4], fromto[5]])
            geom.set("fromto", fromto)

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('reacher.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        action_factor = context_kwargs.get("action_factor", None)
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Reacher(target_size=_BIG_TARGET, random=random, action_factor=action_factor)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking', 'easy')
def easy_morph(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    return easy(time_limit, random, environment_kwargs, context_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        action_factor = context_kwargs.get("action_factor", None)
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Reacher(target_size=_SMALL_TARGET, random=random, action_factor=action_factor)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard_morph(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    return hard(time_limit, random, environment_kwargs, context_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""

    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return (self.named.data.geom_xpos['target', :2] -
                self.named.data.geom_xpos['finger', :2])

    def finger_to_target_dist(self):
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
    """A reacher `Task` to reach the target."""

    def __init__(self, target_size, random=None, action_factor=None):
        """Initialize an instance of `Reacher`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target_size = target_size
        self._action_factor = 1.0 if action_factor is None else action_factor
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.named.model.geom_size['target', 0] = self._target_size
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        # Randomize target position
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(.05, .20)
        physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['to_target'] = physics.finger_to_target()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action) * self._action_factor
        physics.set_control(action)
