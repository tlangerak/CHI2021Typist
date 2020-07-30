import sys

sys.path.insert(0, 'C:\\Users\\thoma\PycharmProjects\\fitts_outcome_motor_control\\forces_pro')
import libs_Intel.win64
import forcespro
import forcespro.nlp
import numpy as np
import casadi
import scipy.interpolate
import matplotlib.pyplot as plt
from planner import Planner
from itertools import count
import time


class FingerController:
    def __init__(self):
        self.index = {"x_force": 0,
                      "y_force": 1,
                      "z_force": 2,
                      "theta_acc": 3,
                      "x_position": 4,
                      "y_position": 5,
                      "z_position": 6,
                      "x_velocity": 7,
                      "y_velocity": 8,
                      "z_velocity": 9,
                      "theta_position": 10,
                      "theta_velocity": 11,
                      "px": [0, 1, 2],
                      "py": [3, 4, 5],
                      "pz": [6, 7, 8],
                      "pv": [9, 10, 11],
                      "pdx": [12, 13],
                      "pdy": [14, 15],
                      "pdz": [16, 17],
                      "pdv": [18, 19],
                      "x_radius": 20,
                      "y_radius": 21,
                      "contour_weight": 22,
                      "lag_weight": 23,
                      "xy_input_weight": 24,
                      "z_input_weight": 25,
                      "velocity_weight": 26,
                      "theta_input_weight": 27,
                      "variance_weight": 28
                      }

        self.dt = 1e-2  # timestep
        self.dt2 = self.dt ** 2
        self.m = 0.1  # mass of model
        self.nStages = 1000  # discretatzion size of spline
        self.fitlength = 100  # how many points are use to fit the ax2+bx+c for the mpcc
        self.fitrange = np.zeros(self.fitlength)
        self.ref = np.zeros((self.nStages, 3))  # the reference spline
        self.theta = np.zeros(self.nStages)  # position on spline
        self.n = 1  # index to take output of mpcc from, starts at 1
        self.p = np.zeros((3, 3))
        self.A = np.asarray([
            [1., 0., 0., self.dt, 0., 0., 0, 0],
            [0., 1., 0., 0., self.dt, 0., 0, 0],
            [0., 0., 1., 0., 0., self.dt, 0, 0],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., self.dt],
            [0., 0., 0., 0., 0., 0., 0., 1.],
        ])
        self.B = np.asarray([
            [0.5 * self.dt2 / self.m, 0, 0, 0],
            [0, 0.5 * self.dt2 / self.m, 0, 0],
            [0, 0, 0.5 * self.dt2 / self.m, 0],
            [self.dt / self.m, 0, 0, 0],
            [0, self.dt / self.m, 0, 0],
            [0, 0, self.dt / self.m, 0],
            [0, 0, 0, 0.5 * self.dt2],
            [0, 0, 0, self.dt]
        ])
        self.nx, self.nu = np.shape(self.B)  # number of states, number of inputs
        self.max_motor_units = 10
        self.average_activation = 0.1

        plot_length = 100
        self.data_logger_values = ["contour", "lag", "vel", "var", "input_xy", "input_z", "input_t"]
        self.data_logger = {}
        for name in self.data_logger_values:
            self.data_logger[name] = [0.] * plot_length

    def setup(self, rx, ry, rz, generate=False):
        self.uxmin = [-casadi.inf] * (self.nu + self.nx)
        self.uxmax = [casadi.inf] * (self.nu + self.nx)

        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = 10  # horizon length
        self.model.nvar = self.nx + self.nu  # number of stage variables
        self.model.neq = self.nx  # number of equality constraints
        self.model.npar = 29  # number of runtime parameters

        self.model.objective = self.eval_obj  # eval_obj is a Python function
        self.model.eq = self.eval_dynamics  # handle to inter-stage function
        self.model.E = np.concatenate([np.zeros((self.nx, self.nu)), np.eye(self.nx)], axis=1)  # selection matrix
        self.model.ub = np.asarray(self.uxmax)  # simple upper bounds
        self.model.lb = np.asarray(self.uxmin)
        self.model.xinitidx = list(range(self.nu, self.nx + self.nu))

        if generate:
            codeoptions = forcespro.CodeOptions("test")
            codeoptions.maxit = 100000  # Maximum number of iterations
            codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but not for timings)
            codeoptions.optlevel = 2  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
            codeoptions.nlp.stack_parambounds = True
            codeoptions.nlp.TolStat = 1E-5
            codeoptions.nlp.TolEq = 1E-6
            codeoptions.nlp.TolIneq = 1E-6
            codeoptions.nlp.TolComp = 1E-6
            codeoptions.overwrite = 1
            self.solver = self.model.generate_solver(codeoptions)

        self.reset()

    def reset(self):
        '''
        :return: resets the finger state position to [0]
        '''
        self.compute_refpositions(rx, ry, rz)
        self.solver = forcespro.nlp.Solver.from_directory("test")
        xi = [0] * (self.nx + self.nu)  # assuming lb and ub are numpy arrays
        self.x0 = np.tile(xi, (self.model.N,))
        self.xinit = np.asarray([0] * len(self.model.xinitidx))

    def calculate_mean_variance(self, desired_force):
        desired_impulse = desired_force * self.dt
        average_impulse = self.average_activation * self.dt
        a = desired_impulse / (self.max_motor_units * average_impulse)
        mu = self.max_motor_units * a
        sigma = casadi.sqrt(self.max_motor_units * casadi.fabs(a) *  casadi.fabs(1 -  casadi.fabs(a)))
        return mu, sigma

    def step_sample(self, desired_force):
        '''
        :param desired_force: the output desired force of the mpcc
        :return: a force sample around the desired force. Used to add noise to the system
        '''
        average_impulse = self.average_activation * self.dt
        mu, sigma = self.calculate_mean_variance(desired_force)
        return np.random.normal(mu, sigma) * average_impulse / self.dt

    def eval_dynamics(self, z):
        '''
        :param z: list of [actions, states]
        :return: list of states in the next timestep.
        '''

        x = z[self.nu:]
        u = z[:self.nu]
        return self.A @ x + self.B @ u

    def calculate_tangent(self, dx, dy, dz):
        tangent = casadi.vcat([dx, dy, dz])
        tangent_normalized = tangent / casadi.sqrt(casadi.dot(tangent, tangent))
        return tangent_normalized

    def error_contour_lag(self, z, p, slacked=False):
        theta = z[self.index["theta_position"]]
        px = z[self.index['x_position']]
        py = z[self.index['y_position']]
        pz = z[self.index['z_position']]
        rx = casadi.polyval(p[self.index["px"]], theta)
        ry = casadi.polyval(p[self.index["py"]], theta)
        rz = casadi.polyval(p[self.index["pz"]], theta)
        drx = casadi.polyval(p[self.index["pdx"]], theta)
        dry = casadi.polyval(p[self.index["pdy"]], theta)
        drz = casadi.polyval(p[self.index["pdz"]], theta)

        tangent_normalized = self.calculate_tangent(drx, dry, drz)

        # set point
        s = casadi.vcat([rx, ry, rz])
        pos = casadi.vcat([px, py, pz])
        r = s - pos

        # lag
        lag = casadi.dot(r, tangent_normalized)
        e_lag = lag ** 2

        # contour
        contour = r - (lag * tangent_normalized)
        e_contour = casadi.dot(contour, contour)

        if slacked:
            # pass
            # delta = 0.5
            d_radius = casadi.sqrt(e_contour / p[self.index["x_radius"]] ** 2)-1
            # e_contour = casadi.if_else(d_radius > delta, delta * d_radius ** 2 - 0.5 * delta ** 2, 0.5 * d_radius ** 2)
            e_contour = casadi.if_else(d_radius > 0, d_radius ** 2, 0)

        return e_contour, e_lag

    def error_velocity(self, z, p):
        dpx = z[self.index['x_velocity']]
        dpy = z[self.index['y_velocity']]
        dpz = z[self.index['z_velocity']]
        theta = z[self.index["theta_position"]]
        drx = casadi.polyval(p[self.index["pdx"]], theta)
        dry = casadi.polyval(p[self.index["pdy"]], theta)
        drz = casadi.polyval(p[self.index["pdz"]], theta)
        desired_velocity = casadi.polyval(p[self.index["pv"]], theta)
        tangent_normalized = self.calculate_tangent(drx, dry, drz)
        global_velocity = casadi.vcat([dpx, dpy, dpz])
        projected_velocity = casadi.dot(global_velocity, tangent_normalized)
        e_velocity = (desired_velocity - projected_velocity) ** 2
        return e_velocity

    def error_input(self, z, p):
        input_error_xy = z[self.index["x_force"]] ** 2 + z[self.index["y_force"]] ** 2
        input_error_z = z[self.index["z_force"]] ** 2
        input_error_t = z[self.index["theta_acc"]] ** 2
        return input_error_xy, input_error_z, input_error_t


    def error_variance(self,z, p):
        _, sigma_x = self.calculate_mean_variance(z[self.index["x_force"]])
        _, sigma_y = self.calculate_mean_variance(z[self.index["y_force"]])
        _, sigma_z = self.calculate_mean_variance(z[self.index["z_force"]])
        return sigma_x**2+sigma_y**2+sigma_z**2

    def eval_obj(self, z, p):
        '''
        :param z: list of [actions, states]
        :param p: parameters
        :return: the cost of taking an action in a state.
        '''

        e_contour, e_lag = self.error_contour_lag(z, p, slacked=True)
        e_velocity = self.error_velocity(z, p)
        e_variance = self.error_variance(z,p)
        e_input_xy, e_input_z, e_input_t = self.error_input(z, p)
        errors = [e_contour, e_lag, e_velocity, e_variance, e_input_xy, e_input_z, e_input_t]

        for name, error in zip(self.data_logger_values, errors):
            self.update_data_logger(error, name)

        return p[self.index["contour_weight"]] * e_contour + \
               p[self.index["lag_weight"]] * e_lag + \
               p[self.index["velocity_weight"]] * e_velocity + \
               p[self.index["variance_weight"]] * e_variance + \
               p[self.index["xy_input_weight"]] * e_input_xy + \
               p[self.index["z_input_weight"]] * e_input_z + \
               p[self.index["theta_input_weight"]] * e_input_t

    def update_parameters(self, paras, p, dp):
        paras[self.index["px"]] = p[:, 0]
        paras[self.index["py"]] = p[:, 1]
        paras[self.index["pz"]] = p[:, 2]
        paras[self.index["pdx"]] = dp[:, 0]
        paras[self.index["pdy"]] = dp[:, 1]
        paras[self.index["pdz"]] = dp[:, 2]
        paras[self.index["pv"]] = [0, 0, 3e-2]
        paras[self.index["pdv"]] = [0, 0]
        return paras

    def apply_output(self, system_in, add_noise=False):
        if add_noise:
            for i in [self.index["x_force"], self.index["y_force"], self.index["z_force"]]:
                system_in[i] = self.step_sample(system_in[i])
        x = self.eval_dynamics(system_in)
        return x

    def step(self, paras, recalc=True):
        '''
        :param paras: the parameters of the cost function
        :param recalc: update the mpcc if true
        :return: applies output of mpcc (and recalculates it) and noise to a system state
        '''

        p, dp = self.fit_parabola(self.xinit[6])
        self.p = p
        paras = self.update_parameters(paras, p, dp)
        parameters = np.tile(paras, self.model.N)
        if recalc:
            self.n = 1
            problem = {"x0": self.x0,
                       "xinit": self.xinit,
                       "all_parameters": parameters
                       }
            self.output, exitflag, info = self.solver.solve(problem)

            assert exitflag == 1, "Some issue with FORCES solver. Exitflag: {}".format(exitflag)
        else:
            self.n += 1
        # print("solver", info.solvetime)
        ux = 'x' + str(self.n).zfill(2)
        mpcc_output = self.output[ux]
        self.output[ux][self.nu:] = self.apply_output(mpcc_output, add_noise=True)
        self.eval_obj(self.output[ux], parameters)
        self.xinit = self.output[ux][self.nu:]
        return self.output[ux][self.nu:]

    def compute_refpositions(self, x, y, z):
        '''
        :param x: ordered list of x-coordinates
        :param y: ordered list of y-coordinates
        :param z: ordered list of z-coordinates
        :return: xyz-coordinates of discretized (by Nstages) spline fited through input with corresponding theta coordinates
        '''

        keyframes = np.column_stack((x, y, z)).T
        dif = np.diff(keyframes, n=1, axis=1).T
        dif = np.power(np.dot(np.power(dif, 2), np.ones((3, 1))), 0.25)
        dif = np.insert(dif, 0, 0., axis=0)
        theta_of_keyframes = np.cumsum(dif).T
        theta = np.linspace(theta_of_keyframes[0], theta_of_keyframes[-1], self.nStages)
        self.ref[:, 0] = scipy.interpolate.pchip_interpolate(theta_of_keyframes, keyframes[0, :], theta)
        self.ref[:, 1] = scipy.interpolate.pchip_interpolate(theta_of_keyframes, keyframes[1, :], theta)
        self.ref[:, 2] = scipy.interpolate.pchip_interpolate(theta_of_keyframes, keyframes[2, :], theta)
        self.theta = theta

    def fit_parabola(self, t):
        '''
        :param t: current theta
        :return: parameters of the spline fitted through s(theta) and its derivative.
        '''
        theta_distance = np.abs(self.theta - t)
        idx = list(theta_distance).index(min(theta_distance))

        if idx < 0.5 * self.fitlength:
            fitrange = np.linspace(0, self.fitlength, self.fitlength + 1).astype(int)
        elif idx > self.nStages - 0.5 * self.fitlength - 1:
            fitrange = np.linspace(self.nStages - 1, self.nStages - self.fitlength - 1, self.fitlength + 1).astype(int)
        else:
            fitrange = np.linspace((idx - 0.5 * self.fitlength), idx + 0.5 * self.fitlength, self.fitlength + 1).astype(
                int)
        self.fitrange = fitrange

        p = np.zeros((3, 3))
        dp = np.zeros((2, 3))

        p[:, 0] = np.polyfit(self.theta[fitrange], self.ref[fitrange, 0], 2)
        p[:, 1] = np.polyfit(self.theta[fitrange], self.ref[fitrange, 1], 2)
        p[:, 2] = np.polyfit(self.theta[fitrange], self.ref[fitrange, 2], 2)
        dp[:, 0] = np.polyder(p[:, 0])
        dp[:, 1] = np.polyder(p[:, 1])
        dp[:, 2] = np.polyder(p[:, 2])

        return p, dp

    def update_data_logger(self, value, name):
        self.data_logger[name].append(float(value))
        self.data_logger[name].pop(0)


if __name__ == '__main__':
    finger = FingerController()
    planner = Planner()

    s = [0, 0, 0]
    m = 1e-2
    e = [0.00, 2e-2, 0]
    t = [0, finger.nStages / 2, finger.nStages]

    # now we create a parabola, to get poitns from, to fit to. This is cumbersome, but easier if change trajectories in the future.
    px, py, pz = planner.create_parabola(s, m, e, t)
    rx, ry, rz = planner.create_parabola_points(1000, px, py, pz)

    plotter = False
    if plotter:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        fig_e = plt.figure(2)
        ax_c = fig_e.add_subplot(411)
        ax_l = fig_e.add_subplot(412)
        ax_t = fig_e.add_subplot(413)
        ax_i = fig_e.add_subplot(414)
        ax_e = [ax_c, ax_l, ax_t, ax_i]

    finger.setup(rx, ry, rz, False)
    paras = np.zeros(finger.model.npar)
    paras[finger.index["x_radius"]] = 3e-3
    paras[finger.index["y_radius"]] = 2e-3
    paras[finger.index["contour_weight"]] = 1e2
    paras[finger.index["lag_weight"]] = 1e5
    paras[finger.index["xy_input_weight"]] = 1e-1
    paras[finger.index["z_input_weight"]] = 1e-1
    paras[finger.index["theta_input_weight"]] = 1e-10
    paras[finger.index["velocity_weight"]] = 1e1
    paras[finger.index["variance_weight"]] = 0

    xsave = []
    ysave = []

    for n in range(100):
        finger.reset()
        print(n)
        for i in count():

            xinit = finger.step(paras, recalc=True)
            if plotter:
                t1 = xinit[-2]

                rxt = np.polyval(paras[finger.index["px"]], t1)
                ryt = np.polyval(paras[finger.index["py"]], t1)
                rzt = np.polyval(paras[finger.index["pz"]], t1)

                pxt = []
                pyt = []
                pzt = []
                for it in np.linspace(t1 - 5, t1 + 5, 20):
                    pxt.append(np.polyval(paras[finger.index["px"]], it))
                    pyt.append(np.polyval(paras[finger.index["py"]], it))
                    pzt.append(np.polyval(paras[finger.index["pz"]], it))

                s = time.time()
                ax.cla()
                [a.cla() for a in ax_e]

                ax_c.plot(finger.data_logger["contour"])
                ax_l.plot(finger.data_logger["lag"])
                ax_t.plot(finger.data_logger["vel"])
                ax_i.plot(finger.data_logger["input_xy"])
                ax_i.plot(finger.data_logger["input_z"])
                ax_i.plot(finger.data_logger["input_t"])

                ax.scatter([xinit[0]], [xinit[1]], [xinit[2]])
                ax.scatter([rxt], [ryt], [rzt])
                ax.plot(finger.ref[:, 0], finger.ref[:, 1], finger.ref[:, 2])
                ax.plot(pxt, pyt, pzt)

                ax.set_ylim([0, 0.02])
                ax.set_xlim([-0.1, 0.1])
                ax.set_zlim([-0.02, 0.02])

                plt.draw()
                plt.pause(0.000001)
                e = time.time()
            if xinit[2] < 0 and xinit[-2] >= finger.theta[-1]:
                xsave.append(xinit[0])
                ysave.append(xinit[1])
                break
    plt.scatter(xsave, ysave)
    plt.show()