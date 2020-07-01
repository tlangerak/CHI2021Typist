import numpy as np
import matplotlib.pyplot as plt
from MotorPool.MotorUnit import MotorUnit


class MotorPool:
    def __init__(self, sz):
        self.sample_time = sz  # s
        self.max_motor_units = 25000
        self.desired_force = 0  # N
        self.current_force = 0
        self.motor_units = [MotorUnit() for _ in range(self.max_motor_units)]

        self.a = (self.desired_force * self.sample_time) / (self.max_motor_units * self.motor_units[0].impulse)
        # thres = 0
        # n = self.max_motor_units
        # for k in range(self.max_motor_units):
        #     b = scipy.special.binom(n, k)
        #     thres += b * self.a ** (n - k) * (1 - self.a) ** k
        # self.thres = thres

    def set_force(self, f):
        self.desired_force = abs(f)  # N
        self.a = (self.desired_force * self.sample_time) / (self.max_motor_units * self.motor_units[0].impulse)
        # thres = 0
        # n = self.max_motor_units
        # for k in range(self.max_motor_units):
        #     b = scipy.special.binom(n, k)
        #     thres += b * self.a ** (n - k) * (1 - self.a) ** k
        # self.thres = self.a
        # self.motor_units = [MotorUnit() for _ in range(self.max_motor_units)]

    def run(self, runtime=3):
        current_time = 0
        force = []
        sf = []
        while current_time < runtime:
            # lf = self.step(current_time)
            current_time += self.sample_time
            force.append(self.current_force)
            sf.append(self.step_sample())
        return force, sf

    def step(self, current_time):
        lf = 0
        for mu in self.motor_units:
            if not mu.activated:
                toss = np.random.random()
                if toss <= self.a:
                    mu.start(current_time)

            if mu.activated:
                f = mu.step(current_time)
                lf += f
        self.current_force = lf
        return lf

    # def stepv2(self, current_time):
    #     lf = 0
    #     for mu in self.motor_units:
    #         if not mu.activated:
    #             toss = np.random.random()
    #             if toss <= self.thres:
    #                 mu.start(current_time)
    #         if mu.activated:
    #             f = mu.step(current_time)
    #             lf += f
    #     self.current_force = lf
    #     return lf


    def step_sample(self):
        a = self.desired_force / (self.max_motor_units * self.motor_units[0].average_activation)
        # a=self.a
        return np.random.normal(self.max_motor_units * a,
                                np.sqrt(self.max_motor_units * a * (1-a))) * self.motor_units[0].average_activation


if __name__ == '__main__':

    x = []
    y = []
    ys = []
    fd = []
    ysvar = []
    yvar = []
    t = np.arange(0, 1, 0.01)
    for i in t:
        for n in range(5):
            mp = MotorPool(1e-3)
            mp.set_force(i)
            f, fs = mp.run(1.5)
            f = f[500:1000]
            fs = fs[500:1000]
            x.append(i)
            fd.append(i)
            y.append(np.mean(f))
            ys.append(np.mean(fs))
            yvar.append(np.var(f))
            ysvar.append(np.var(fs))
            print(n, i)

    plt.figure(1)
    plt.plot(t, t)
    plt.scatter(x, y, label="orginal")
    plt.scatter(x, ys, label="sample")
    plt.legend()

    plt.figure(2)
    plt.scatter(x, yvar, label="orginal")
    plt.scatter(x, ysvar, label="sample")
    plt.xlabel("Set Force (N)")
    plt.ylabel("Variance (N)")
    plt.legend()
    plt.show()
