# https://github.com/seba-1511/shapechanger/tree/master/mj_transfer

import numpy as np
import torch
import gym
import custom_envs
from itertools import count
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
ENV = 'Finger-v1'

if __name__ == '__main__':
    num_cpu = 4
    # env = SubprocVecEnv([make_env(ENV, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(ENV, n_envs=num_cpu, seed=0)

    env = gym.make(ENV)
    model = SAC(MlpPolicy, env, verbose=2, tensorboard_log="./SAC_finger")
    model.learn(total_timesteps=30000)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()