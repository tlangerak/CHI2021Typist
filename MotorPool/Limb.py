from MotorPool.MotorPool import MotorPool
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


class Limb:
    def __init__(self):
        self.current_time = 0.
        self.stepsize = 1e-3
        self.agonist = MotorPool(self.stepsize)
        self.antagonist = MotorPool(self.stepsize)
        self.agonist.set_force(0)
        self.antagonist.set_force(0)
        self.AgoF = 0
        self.AntF = 0
        self.I = 0.1  # kg
        self.Eps = 0.5
        self.X_max = 1.  # m
        self.X = -0.08  # m
        self.dX = 0  # m/s
        self.ddX = 0  # m/s^2
        self.Fs = 0.
        self.Fd = 0.
        self.Gk = 1

    def calculate_dynamic_force(self, AgoF, AntF):
        return AgoF - AntF

    def calculate_static_force(self, AgoF, AntF):
        return AgoF + AntF

    def step(self):
        self.AgoF = self.agonist.step(self.current_time)
        self.AntF = self.antagonist.step(self.current_time)
        self.Fs = self.calculate_static_force(self.AgoF, self.AntF)
        self.Fd = self.calculate_dynamic_force(self.AgoF, self.AntF)
        self.ddX = self.Fd / self.I
        self.dX += self.ddX * self.stepsize
        self.X += self.dX * self.stepsize + self.ddX * self.stepsize ** 2
        self.current_time += self.stepsize

    def stepv2(self):
        self.AgoF = self.agonist.step(self.current_time)
        self.AntF = self.antagonist.step(self.current_time)

        self.AgoF = limb.agonist.desired_force
        self.AntF = limb.antagonist.desired_force*-1

        self.Fs = self.calculate_static_force(self.AgoF, self.AntF)
        self.Fd = self.calculate_dynamic_force(self.AgoF, self.AntF)

        sol = optimize.least_squares(self.solve_position, np.asarray([self.X]))
        x = sol.x[0]
        dx = x - self.X
        ddx = dx - self.dX
        self.X = x
        self.dX = dx
        self.ddX = ddx
        self.current_time += self.stepsize

    def solve_position(self, x):
        K = self.Gk * self.Fs
        B = 2 * self.Eps * np.sqrt(abs(self.I * K))
        Gf  = 1
        force = Gf * self.Fd
        dX = (x - self.X) / self.stepsize
        ddX = (dX - self.dX) / self.stepsize
        pos = self.I * ddX * self.stepsize ** 2 + B * dX * self.stepsize + K * x
        return pos - force

    def stepv3(self):
        self.AgoF = self.agonist.step(self.current_time)
        self.AntF = self.antagonist.step(self.current_time)
        print(self.AgoF, self.AntF)
        self.Fs = self.calculate_static_force(self.AgoF, self.AntF)*0.1
        self.Fd = self.calculate_dynamic_force(self.AgoF, self.AntF)*0.1
        self.ddX = self.Fd / self.I
        self.dX += self.ddX * self.stepsize
        K = self.Gk * self.Fs
        B = 2 * self.Eps * np.sqrt(abs(self.I * K))
        self.X = self.ddX * self.stepsize ** 2 + B*self.dX * self.stepsize + self.X
        self.current_time += self.stepsize


if __name__ == '__main__':
    for j in range(15):
        limb = Limb()
        x = []
        dx = []
        f = []
        fd = []
        fa = []
        for i in range(500):
            if i < 100:
                limb.agonist.set_force(35)
                limb.antagonist.set_force(-45)
            elif i >= 100 and i < 190:
                limb.agonist.set_force((i - 100) * (65 - 35) / (190 - 100) + 35)
                limb.antagonist.set_force((i - 100) * (45 - 15) / (190 - 100) - 45)
            elif i >= 190 and i < 400:
                limb.agonist.set_force((i - 190) * (-65) / (400 - 190) + 65)
                limb.antagonist.set_force((i - 190) * (-50) / (400 - 190) - 15)
            else:
                limb.antagonist.set_force(0)
                limb.agonist.set_force(0)

            limb.stepv3()
            x.append(limb.X)
            dx.append(limb.dX)
            f.append(limb.antagonist.desired_force*-1*0.1)
            fd.append(limb.agonist.desired_force*0.1)
            fa.append((limb.antagonist.current_force*0.1*-1))

        plt.figure(1)
        plt.plot(x)
        plt.figure(3)
        plt.plot(dx)
        plt.figure(2)
        plt.plot(f)
        plt.plot(fd)
        plt.figure(4)
        plt.plot(f)
        plt.plot(fa)
    plt.show()
