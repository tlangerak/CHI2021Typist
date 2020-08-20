import numpy as np
import gym
import custom_envs
ENV = 'Finger-v1'

if __name__ == '__main__':
    env = gym.make(ENV)
    for i in range(15):
        s = env.reset_mpcc()
        a = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        while True:
            print("R")
            env.reset_mpcc()
            # print(env.get_body_com("target"))
            # print(env.action_space.shape)
            a = (np.random.rand(*env.action_space.shape) - 0.5) * 1.1
            s, r, d, _ = env.step(a)
            # # print('Reward: ', r)
            env.render()
            # # sleep(0.1)
            # # a += 0.1
            # if d:
            #     break