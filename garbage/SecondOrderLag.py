from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import count


class Agent:
    def __init__(self):
        # States
        self.state0 = [0, 0, 0, 0, 0.1, 0]
        self.start = [0, 0, 0]

        # Actions
        self.u = [0, 0, 0]
        self.k = [1, 1, 1]
        self.d = 0.35
        self.m = 0.05

        # User Parameters
        self.a = 0.1
        self.b = 0.2
        self.x_offset = 0
        self.y_offset = 0
        self.thres = 0.5e-3

        # Timing
        self.timer = 0
        self.ti = 0.0  # initial time
        self.tf = 0.01  # final time
        self.step_size = 0.001  # step
        self.t = np.arange(self.ti, self.tf, self.step_size)
        self.update_acc = lambda m, u, d, xd, k, x: (1 / m) * (u - d * xd - k * x)

    def update_target(self, target):
        self.u[:2] = target

    def model(self, state, t, u, k, d, m):
        kx, ky, kz = k  # spring constant, N/m <-- seems dependant on ID
        x, xd, y, yd, z, zd = state  # x,y position and velocity of end effector
        ux, uy, uz = u  # x,y psotion of target
        xdd = self.update_acc(m, ux, d, xd, kx, x)
        ydd = self.update_acc(m, uy, d, yd, ky, y)
        zdd = self.update_acc(m, uz, d, zd, kz, z)
        return [xd, xdd, yd, ydd, zd, zdd]

    def sample(self, sd, n_samples=100):
        return [np.random.normal(0, sd) for _ in range(n_samples)]

    def step(self, u=None, k=None, d=None, m=None):
        if u is None:
            u = self.u
        if k is None:
            k = self.k
        if d is None:
            d = self.d
        if m is None:
            m = self.m

        distance_to_go = ((self.state0[0] - self.u[0]) ** 2 + (self.state0[2] - self.u[1]) ** 2) ** 0.5
        self.u[2] = distance_to_go
        state = odeint(self.model, self.state0, self.t, args=(u, k, d, m,))
        self.state0 = state[-1]
        self.timer += self.step_size

        if self.state0[4] < 1e-3:
            x_sample, y_sample = self.movement_stop()
            state[-1][0] = x_sample[0]
            state[-1][2] = y_sample[2]
            return state[-1], True
        else:
            return state[-1], False

    def movement_stop(self):
        Ax = ((self.start[0] - self.state0[0]) ** 2) ** 0.5
        Ay = ((self.start[1] - self.state0[2]) ** 2) ** 0.5
        MT = self.timer * (self.tf / self.step_size)

        SDx = Ax / ((np.sqrt(2 * np.pi * np.e) * (2 ** ((MT - self.a) / self.b) - 1)))
        SDy = Ay / ((np.sqrt(2 * np.pi * np.e) * (2 ** ((MT - self.a) / self.b) - 1)))
        x_sample = [s + self.state0[0] + self.x_offset for s in self.sample(SDx)]
        y_sample = [s + self.state0[2] + self.y_offset for s in self.sample(SDy)]
        self.timer = 0
        self.state0[0] = x_sample[0]
        self.state0[2] = y_sample[0]
        return x_sample, y_sample


if __name__ == '__main__':
    model = Agent()
    model.u = [0.15, 0.3, 0]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig2, ax3 = plt.subplots()
    x = []
    y = []
    z = []
    xd = []
    yd = []
    zd = []
    ux = []
    uy = []
    uz = []
    start = time.time()
    done = False
    for t in count():
        if t == 50:
            model.update_target([0.33, 0.1])
        state, done = model.step()
        x.append(state[0])
        xd.append(state[1])
        y.append(state[2])
        yd.append(state[3])
        z.append(state[4])
        zd.append(state[5])
        ux.append(model.u[0])
        uy.append(model.u[1])
        uz.append(model.u[2])
        if done:
            break


    ax1.plot(ux, 'r+', linewidth=0.5)
    ax1.plot(uy, 'g+', linewidth=0.5)
    ax1.plot(uz, 'b+', linewidth=0.5)
    ax1.plot(x, 'r', label=r'$x (m)$', linewidth=1.0)
    ax2.plot(xd, 'r--', label=r'$\dot{x} (m/sec)$', linewidth=1.0)
    ax1.plot(y, 'g', label=r'$y (m)$', linewidth=1.0)
    ax2.plot(yd, 'g--', label=r'$\dot{y} (m/sec)$', linewidth=1.0)
    ax1.plot(z, 'b', label=r'$z (m)$', linewidth=1.0)
    ax2.plot(zd, 'b--', label=r'$\dot{z} (m/sec)$', linewidth=1.0)
    ax3.plot(x, y)
    ax2.legend(loc='lower right')
    ax1.legend()
    ax1.set_xlabel('step')
    ax1.set_ylabel('disp (mm)', color='b')
    ax2.set_ylabel('velocity (m/s)', color='g')
    plt.show()
