import matplotlib.pyplot as plt


class MotorUnit:
    def __init__(self):
        self.start_time = 0 #s
        self.activation_duration = 25e-3 # s
        self.average_activation = 0.25 # N
        self.impulse = self.average_activation * self.activation_duration #Ns
        self.activated = False

    def start(self, current_time):
        self.start_time = current_time
        self.activated = True

    def stop(self):
        self.activated = False

    def step(self, current_time):
        if current_time - self.start_time > self.activation_duration:
            self.activated = False
            return False

        if self.activated:
            force = self.calculate_activation(current_time)
            return force

    def calculate_activation(self, ct):
        t = ct - self.start_time
        if t < 0:
            return 0
        if t < self.activation_duration / 2:
            f = t * ((2 * self.average_activation) / (self.activation_duration / 2))
        else:
            f = (2 * self.average_activation) - (t - self.activation_duration / 2) * (
                        (2 * self.average_activation) / (self.activation_duration / 2))
        return f


# if __name__ == '__main__':
#     mu = MotorUnit()
#     mu.start(0.03)
#     forces = []
#     for i in range(100):
#         forces.append(mu.step(i / 1000.))
#
#     plt.plot(forces)
#     plt.show()
