# This environment is created by Karen Liu (karen.liu@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import pydart2 as pydart


class DartWAMReacherEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        n_dof = 7
        obs_dim = 27
        frame_skip = 4
        self.fingertip = np.array([0.0, 0.0, 0.3])  # z direction of the last link
        self.target = np.array([0.6, 0.6, 0.6])
        self.action_scale = np.array([1.8, 1.8, 1.8, 1.6, 0.6, 0.6, 0.174]) * 10.0
        self.control_bounds = np.vstack([np.ones(n_dof), -np.ones(n_dof)])
        super().__init__(['wam/reaching_target.skel', 'wam/wam.urdf'],
                         frame_skip, obs_dim, self.control_bounds, disableViewer=False)
        # self.robot_skeleton.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)  # lock the base of wam
        utils.EzPickle.__init__(self)

    def step(self, a):
        # print(dir(self.viewer.scene.cameras[0]))
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        vec = self.robot_skeleton.bodynodes[-1].to_world(self.fingertip) - self.target
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(tau).sum() * 0.001
        alive_bonus = 0
        reward = reward_dist + reward_ctrl + alive_bonus

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (-reward_dist > 0.1))
        # print(self.robot_skeleton.bodynodes[0].com())

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        vec = self.robot_skeleton.bodynodes[-1].to_world(self.fingertip) - self.target
        return np.concatenate([np.cos(theta), np.sin(theta), self.target, self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        while True:
            self.target = self.np_random.uniform(low=-1, high=1, size=3)
            if np.linalg.norm(self.target) < 1.5 and np.linalg.norm(self.target) > 0.6:
                break
        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
