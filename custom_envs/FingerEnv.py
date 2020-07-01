import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


class FingerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.success_counter = 1
        self.step_counter = 0
        utils.EzPickle.__init__(self)
        env_file = 'C:\\Users\\thoma\PycharmProjects\\fitts_outcome_motor_control\custom_envs\\finger.xml'
        mujoco_env.MujocoEnv.__init__(self, env_file, 2)
        self.state = self._get_obs()
        self.reset()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        distance = np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = - distance + reward_ctrl

        ob = self._get_obs()
        done = distance < 0.002
        # done = False
        return ob, reward, done, dict(reward_dist=-distance, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2

    def reset_model(self):
        self.step_counter = 1
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) +self.init_qpos

        qpos = self.init_qpos
        while True:
            self.goal = (self.np_random.uniform(low=-1.5, high=-0.5, size=1),
                         self.np_random.uniform(low=-0.3, high=0.3, size=1),
                         self.np_random.uniform(low=-1.8, high=0.1, size=1))
            if np.linalg.norm(self.goal) < 2: break
        qpos[-3:] = self.goal
        # qpos[-2:] = self.goal
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel = 0.0 * self.init_qvel
        qvel[-3:] = 0
        # qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("fingertip"),
            self.get_body_com("target"),
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
