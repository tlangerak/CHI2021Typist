import numpy as np
from scipy import linalg
import pandas as pd


def system_dynamic(B, A, x, u, y):

    return A.dot(x) + B.dot(u) - y


if __name__ == '__main__':
    # df = pd.read_csv("data_finger_v0.csv")
    # df['split'] = np.random.randn(df.shape[0], 1)
    # msk = np.random.rand(len(df)) <= 0.7
    dt = 0.001
    dt2 = dt ** 2
    n_states = 9
    n_actions = 7
    A = np.asarray([
        [1, 0, 0, dt, 0, 0, dt2, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, dt2, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, dt2],
        [0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    B = np.ones((n_states, n_actions))
    x = np.ones((n_states, 1))
    u = np.zeros((n_actions, 1))
    y = x
    args = {"A": A, "x": x, "u": u, "y": y}
    print(system_dynamic(B, **args))
