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

"""Cheetah Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def update_physics(xml_string, length=1.0, action_factor=1.0):
    """
    Adapts and returns the xml_string of the model with the given context.
    Inspired by
    https://github.com/automl/CARL/blob/main/carl/envs/dmc/dmc_tasks/utils.py
    """

    mjcf = etree.fromstring(xml_string)

    # Update length
    geoms = mjcf.findall(".//geom")
    for geom in geoms:
        if geom.get("name") in ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]:
            pos = geom.get("pos").split(" ")
            pos = " ".join([str(float(pos[0]) * length), pos[1], str(float(pos[2]) * length)])
            geom.set("pos", pos)
            size = geom.get("size").split(" ")
            size = " ".join([size[0], str(float(size[1]) * length)])
            geom.set("size", size)
    bodies = mjcf.findall(".//body")
    for body in bodies:
        if body.get("name") in ["bshin", "bfoot", "fshin", "ffoot"]:
            pos = body.get("pos").split(" ")
            pos = " ".join([str(float(pos[0]) * length), pos[1], str(float(pos[2]) * length)])
            body.set("pos", pos)

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('cheetah.xml'), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    """Returns the run task."""
    xml_string, assets = get_model_and_assets()
    if context_kwargs is not None:
        action_factor = context_kwargs.get("action_factor", None)
        xml_string = update_physics(xml_string=xml_string, **context_kwargs)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Cheetah(random=random, action_factor=action_factor)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def run_morph(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, context_kwargs=None):
    return run(time_limit, random, environment_kwargs, context_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def __init__(self, random=None, action_factor=None):
        """Initialize an instance of `Cheetah`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._action_factor = 1.0 if action_factor is None else action_factor
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        physics.step(nstep=200)

        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return rewards.tolerance(physics.speed(),
                                 bounds=(_RUN_SPEED, float('inf')),
                                 margin=_RUN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action) * self._action_factor
        physics.set_control(action)
