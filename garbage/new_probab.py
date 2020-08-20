import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

dt = 1e-2
max_motor = 500
average_force = 0.1
average_impulse = average_force * dt


def n_activated(df, activated_motors):
    desired_impulse = df * dt
    p_activation = desired_impulse / (max_motor * average_impulse)
    p_not_busy = 1 - (activated_motors / max_motor)
    mu = max_motor * p_activation * p_not_busy
    sigma = np.sqrt(max_motor * abs(p_not_busy * p_activation) * abs(1 - abs(p_not_busy * p_activation)))
    return int(np.random.normal(mu, sigma))


def n_activated_v2(df, activated_motors):
    desired_impulse = df * dt
    desired_motors = (desired_impulse / average_impulse)
    desired_new_motors = max(0, desired_motors - activated_motors)
    a = desired_new_motors / max_motor
    p = (1 - (activated_motors / max_motor)) * a
    sigma = np.sqrt(max_motor * p * (1 - p))
    return int(np.random.normal(desired_new_motors, sigma))


if __name__ == '__main__':
    desired_forces = range(0, 50, 5)
    average_forces = []
    dfs = []
    for j in range(10):
        for df in desired_forces:
            active = [0] * 25
            force = []
            for i in count():
                active.append(n_activated_v2(df, np.sum(active)))
                active.pop(0)
                force.append(np.sum(active) * average_force)
                if i * dt > 1:
                    break
            average_forces.append(np.std(force))
            dfs.append(df)

    plt.scatter(dfs, average_forces)
    plt.show()
