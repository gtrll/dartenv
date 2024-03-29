import numpy as np
from gym import utils
from gym.envs.dart import dart_env

# TODO FIX the hard coded horizon.


class DartCartPoleWithModelEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0], [-1.0]])
        self.action_scale = 100
        dart_env.DartEnv.__init__(self, 'cartpole.skel', 2, 4, control_bounds, dt=0.02, disableViewer=True)
        utils.EzPickle.__init__(self)

    def set_dynamics(self, dynamics):
        self._dynamics = dynamics

    def _step(self, a):
        reward = 1.0

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        # self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()
        # Using a!!!!!
        ob_next = self._dynamics(np.hstack([ob[None], a[0][None][None]]))[0]
        q_dim = int(len(ob_next)/2)
        # self.robot_skeleton.q = ob_next[:q_dim]
        # self.robot_skeleton.dq = ob_next[q_dim:]
        self.set_state(ob_next[:q_dim], ob_next[q_dim:])
        ob = ob_next

        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
