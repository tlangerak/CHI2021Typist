import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import count


class Plotter:
    def __init__(self, n=4, plot_length=100, h=10):
        s = min(n * 2, 12)
        self.fig = plt.figure(figsize=(s, s), constrained_layout=True)
        gs = self.fig.add_gridspec(2 * n - 1, n)
        ax3d = self.fig.add_subplot(gs[:n, :], projection='3d')
        ax = [ax3d]
        for i in range(n - 1):
            ax.append(self.fig.add_subplot(gs[i + n, :]))

        self.ax = ax
        self.x = list(range(-plot_length, 0))
        self.h = list(range(h))
        self.pos = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.vel = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.force = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.pos_predict = [[0] * h, [0] * h, [0] * h]
        self.vel_predict = [[0] * h, [0] * h, [0] * h]
        self.force_predict = [[0] * h, [0] * h, [0] * h]
        self.tp = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.tv = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.tf = [[0] * plot_length, [0] * plot_length, [0] * plot_length]
        self.set_default()
        plt.draw()
        plt.pause(0.01)

    def set_default(self):
        titles = ['3D', 'position (m)', 'velocity (m/s)', 'force(N)']
        limit = 0.05
        for f, (axs, t) in enumerate(zip(self.ax, titles)):
            axs.set_title(t)
            axs.set_ylim([-limit*10**(f-1), limit*10**(f-1)])

        self.ax[0].set_xlim([-limit, limit])
        self.ax[0].set_ylim([-limit, limit])
        self.ax[0].set_zlim([-limit, limit])

    def update(self, x, tp, tv, tf, position, velocity, force, pos_predict=None, vel_predict=None, force_predict=None):
        self.x.append(x)
        self.x.pop(0)
        self.h.append(self.h[-1] + 1)
        self.h.pop(0)
        for new, l in zip([position, velocity, force, tp, tv, tf],
                          [self.pos, self.vel, self.force, self.tp, self.tv, self.tf]):
            for n in range(3):
                l[n].append(new[n])
                l[n].pop(0)

        if pos_predict is not None:
            self.pos_predict = pos_predict
            self.vel_predict = vel_predict
            self.force_predict = force_predict

    def redraw(self):
        for a in self.ax:
            a.cla()
        self.set_default()

        self.ax[0].scatter([self.pos[0][0]], [self.pos[1][0]], [self.pos[2][0]])

        for data, axis in zip([self.pos, self.vel, self.force], [self.ax[1], self.ax[2], self.ax[3]]):
            axis.plot(self.x, data[0], 'r', label='x')
            axis.plot(self.x, data[1], 'g', label='y')
            axis.plot(self.x, data[2], 'b', label='z')

        for data, axis in zip([self.tp, self.tv, self.tf], [self.ax[1], self.ax[2], self.ax[3]]):
            axis.plot(self.x, data[0], 'r+', label='x target')
            axis.plot(self.x, data[1], 'g+', label='y target')
            axis.plot(self.x, data[2], 'b+', label='z target')

        for data, axis in zip([self.pos_predict, self.vel_predict, self.force_predict],
                              [self.ax[1], self.ax[2], self.ax[3]]):
            axis.plot(self.h, data[0], 'r--', label='x predict')
            axis.plot(self.h, data[1], 'g--', label='y predict')
            axis.plot(self.h, data[2], 'b--', label='z predict')

        plt.draw()
        plt.pause(0.0001)
