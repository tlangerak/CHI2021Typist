import sys

sys.path.insert(0, 'C:\\Users\\thoma\PycharmProjects\\fitts_outcome_motor_control\\forces_pro')
import forcespro
import forcespro.nlp
import numpy as np
import casadi

import matplotlib.pyplot as plt


class FingerController:
    def __init__(self, generate=False):
        self.index = {"x_force": 0,
                      "y_force": 1,
                      "z_force": 2,
                      "x_position": 3,
                      "y_position": 4,
                      "z_position": 5,
                      "x_velocity": 6,
                      "y_velocity": 7,
                      "z_velocity": 8,
                      "x_target": 0,
                      "y_target": 1,
                      "z_target": 2,
                      "x_radius": 3,
                      "y_radius": 4,
                      "xy_position_weight": 5,
                      "xy_input_weight": 6,
                      "z_position_weight": 7,
                      "z_input_weight": 8
                      }

        self.dt = 1e-2
        self.dt2 = self.dt ** 2
        self.m = 0.1

        self.A = np.asarray([
            [1., 0., 0., self.dt, 0., 0.],
            [0., 1., 0., 0., self.dt, 0.],
            [0., 0., 1., 0., 0., self.dt],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.],
        ])
        self.B = np.asarray([
            [0.5 * self.dt2 / self.m, 0, 0],
            [0, 0.5 * self.dt2 / self.m, 0],
            [0, 0, 0.5 * self.dt2 / self.m],
            [self.dt / self.m, 0, 0],
            [0, self.dt / self.m, 0],
            [0, 0, self.dt / self.m],
        ])
        self.nx, self.nu = np.shape(self.B)
        self.max_motor_units = 10
        self.average_activation = 0.1

        self.uxmin = [-10, -10, -10, -0.1, -0.1, -0.1, -10, -10, -10]
        self.uxmax = [10, 10, 10, 0.1, 0.1, 0.1, 10, 10, 10]

        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = 99  # not required if already specified in initializer
        self.model.nvar = self.nx + self.nu  # number of stage variables
        self.model.neq = self.nx  # number of equality constraints
        self.model.npar = 9  # number of runtime parameters

        self.model.objective = self.eval_obj  # eval_obj is a Python function
        self.model.eq = self.eval_dynamics  # handle to inter-stage function
        self.model.E = np.concatenate([np.zeros((self.nx, self.nu)), np.eye(self.nx)], axis=1)  # selection matrix
        self.model.ub = np.asarray(self.uxmax)  # simple upper bounds
        self.model.lb = np.asarray(self.uxmin)
        self.model.xinitidx = list(range(self.nu, self.nx + self.nu))

        codeoptions = forcespro.CodeOptions("test")
        codeoptions.maxit = 250  # Maximum number of iterations
        codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but not for timings)
        codeoptions.optlevel = 2  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        codeoptions.nlp.stack_parambounds = True
        codeoptions.solvemethod = 'SQP_NLP'
        codeoptions.sqp_nlp.maxSQPit = 2

        if generate:
            codeoptions.overwrite = 1
            self.solver = self.model.generate_solver(codeoptions)
        else:
            self.solver = forcespro.nlp.Solver.from_directory("test")

        xi = (finger.model.lb + finger.model.ub) / 2  # assuming lb and ub are numpy arrays
        self.x0 = np.tile(xi, (finger.model.N,))
        self.xinit = np.asarray([0] * len(finger.model.xinitidx))

    def step_sample(self, desired_force):
        desired_impulse = desired_force * self.dt
        average_impulse = self.average_activation * self.dt
        a = desired_impulse / (self.max_motor_units * average_impulse)
        mu = self.max_motor_units * a
        sigma = np.sqrt(self.max_motor_units * abs(a) * abs(1 - abs(a)))
        return np.random.normal(mu, sigma) * average_impulse / self.dt

    def eval_dynamics(self, z):
        x = z[self.nu:]
        u = z[:self.nu]
        return self.A @ x + self.B @ u

    def eval_obj(self, z, p):
        z_position_error = (z[self.index["z_position"]] - p[self.index['z_target']]) ** 2
        z_input_error = (z[self.index["z_force"]]) ** 2
        d_radius = casadi.sqrt(
            (z[self.index['x_position']] - p[self.index['x_target']]) ** 2 / p[self.index["x_radius"]] ** 2 +
            (z[self.index['y_position']] - p[self.index['y_target']]) ** 2 / p[self.index["y_radius"]] ** 2) - 1

        # d_radius = casadi.sqrt((z[index['x_position']] - p[index['x_target']]) ** 2 / p[index["x_radius"]]**2) - 1
        xy_position_error = casadi.if_else(d_radius > 0, d_radius ** 2, 0)
        # xy_position_error = (z[index['x_position']] - p[index['x_target']]) ** 2 + (z[index['y_position']] - p[index['y_target']]) ** 2
        xy_input_error = z[self.index["x_force"]] ** 2 + z[self.index["y_force"]] ** 2
        return p[self.index["xy_position_weight"]] * xy_position_error + \
               p[self.index["z_position_weight"]] * z_position_error + \
               p[self.index["xy_input_weight"]] * xy_input_error + \
               p[self.index["z_input_weight"]] * z_input_error

    def step(self, paras):
        problem = {"x0": self.x0,
                   "xinit": self.xinit
                   }
        parameters = np.tile(paras, self.model.N)
        problem["all_parameters"] = parameters
        output, exitflag, info = self.solver.solve(problem)
        output['x01'][0] = self.step_sample(output['x01'][0])
        output['x01'][1] = self.step_sample(output['x01'][1])
        output['x01'][2] = self.step_sample(output['x01'][2])
        x = self.eval_dynamics(output['x01'])
        assert exitflag == 1, "Some issue with FORCES solver. Exitflag: {}".format(exitflag)

        self.xinit = x

        return x

if __name__ == '__main__':
    finger = FingerController(False)
    # solver.help()

    for e in range(15):
        xsave = []
        paras = np.zeros(finger.model.npar)
        paras[finger.index["x_target"]] = 1e-2
        paras[finger.index["y_target"]] = 0e-2
        paras[finger.index["z_target"]] = 0e-2
        paras[finger.index["x_radius"]] = 5e-3
        paras[finger.index["y_radius"]] = 7e-3
        paras[finger.index["xy_position_weight"]] = 1
        paras[finger.index["xy_input_weight"]] = 0.001
        paras[finger.index["z_position_weight"]] = 1
        paras[finger.index["z_input_weight"]] = 0.001

        for i in range(500):
            xinit = finger.step(paras)
            xsave.append(xinit[0])

        plt.plot(xsave)

    plt.plot([paras[finger.index["x_target"]]] * len(xsave), 'b-')
    plt.plot([paras[finger.index["x_target"]] + paras[finger.index["x_radius"]]] * len(xsave), 'b--')
    plt.plot([paras[finger.index["x_target"]] - paras[finger.index["x_radius"]]] * len(xsave), 'b--')
    plt.show()
