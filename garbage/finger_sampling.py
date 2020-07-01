import gym
import custom_envs
import numpy as np
from itertools import count
import csv
ENV = 'Finger-v1'
import time
if __name__ == '__main__':
    env = gym.make(ENV)

    n_observation = env.observation_space.shape
    n_actions = env.action_space.shape
    print(n_observation, n_actions)
    header = []
    for n in range(7):
        header.append("action_{}".format(n))
    header.append("init_x")
    header.append("init_y")
    header.append("init_z")
    header.append("init_dx")
    header.append("init_dy")
    header.append("init_dz")
    header.append("init_ddx")
    header.append("init_ddy")
    header.append("init_ddz")
    header.append("result_x")
    header.append("result_y")
    header.append("result_z")
    header.append("result_dx")
    header.append("result_dy")
    header.append("result_dz")
    header.append("result_ddx")
    header.append("result_ddy")
    header.append("result_ddz")
    with open('data_finger_v0.csv', 'w', newline='') as f:
        fw = csv.writer(f)
        fw.writerow(header)

    s = env.reset()
    a = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    for epoch in count():
        env.reset()
        # print(env.get_body_com("target"))
        # print(env.action_space.shape)
        data = []
        old_acc = np.zeros(env.get_body_com("fingertip").shape)
        old_vel= np.zeros(env.get_body_com("fingertip").shape)
        new_acc = np.zeros(env.get_body_com("fingertip").shape)
        new_vel = np.zeros(env.get_body_com("fingertip").shape)
        dt =0.001
        for samples in count():
            env._get_obs()
            old_pos = env.get_body_com("fingertip").copy()
            action = (np.random.rand(*env.action_space.shape))
            new_state, r, d, _ = env.step(action)
            # env.render()
            env._get_obs()
            new_pos = env.get_body_com("fingertip").copy()

            datarow = []
            new_vel = (new_pos - old_pos)/dt
            new_acc = (new_acc - old_acc)/dt
            print(new_pos, old_pos)
            datarow.extend(action)
            datarow.extend(old_pos)
            datarow.extend(old_vel)
            datarow.extend(old_acc)
            datarow.extend(new_pos)
            datarow.extend(new_vel)
            datarow.extend(new_acc)
            data.append(datarow)
            old_vel = new_vel
            old_acc = new_acc
            if samples > 5000:
                break
        with open('data_finger_v0.csv', 'a', newline='') as f:
            fw = csv.writer(f)
            fw.writerows(data)

        print(epoch)
        if epoch > 10:
            break