from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from mpcc_forces import FingerController
import numpy as np
from itertools import count
from planner import Planner
import json
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
import dill as pickle
import matplotlib.pyplot as plt

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from scipy.stats import entropy
from scipy.stats import norm

norm.cdf(1.96)


class UserParameters:
    def __init__(self, participant):
        self.finger = FingerController()
        self.finger.setup_mpcc(False)
        self.planner = Planner()
        self.participant = participant
        filen = "fittslaw_simple_p{}.csv".format(participant)
        my_data = np.genfromtxt(filen, delimiter=',')
        self.my_data = my_data[2:]

        self.pixels_per_meter = (900 / 100) * 1e3

        self.pbounds = {
            # 'average_activation': (1e-2, 1e0),
            # 'max_motor_units': (1e0, 1e4),
            # 'w_contour': (1e0, 1e3),
            # 'w_input_t': (1e-4, 1e0),
            # 'w_input_xy': (1e-4, 1e0),
            # 'w_input_z': (1e-4, 1e0),
            # 'w_lag': (1e0, 1e3),
            'w_vel': (1e0, 1e3),
        }
        # self.bounds_transformer = SequentialDomainReductionTransformer()
        self.optimizer = BayesianOptimization(
            f=self.cost,
            pbounds=self.pbounds,
            random_state=1,
            verbose=2,
            # bounds_transformer=self.bounds_transformer
        )

    def cost(self,
             # average_activation,
             # max_motor_units,
             # w_contour,
             # w_input_xy,
             # w_input_z,
             # w_input_t,
             # w_lag,
             w_vel
             ):

        total_error = 0
        for row in self.my_data:
            s = [0, 0, 0]
            m = 1e-2
            e = [row[0], row[1], 0]
            t = [0, self.finger.nStages / 2, self.finger.nStages]

            px, py, pz = self.planner.create_parabola(s, m, e, t)
            rx, ry, rz = self.planner.create_parabola_points(1000, px, py, pz)

            self.finger.reset_mpcc(rx, ry, rz)
            self.finger.update_user_paras(
                300,
                0.5,
                2e3,
                4e3,
                w_vel,
                1e-2,
                1e-2,
                1e-2
            )

            paras = np.zeros(self.finger.model.npar)
            paras[self.finger.index["contour_weight"]] = self.finger.w_contour
            paras[self.finger.index["lag_weight"]] = self.finger.w_lag
            paras[self.finger.index["xy_input_weight"]] = self.finger.w_input_xy
            paras[self.finger.index["z_input_weight"]] = self.finger.w_input_z
            paras[self.finger.index["theta_input_weight"]] = self.finger.w_input_t
            paras[self.finger.index["velocity_weight"]] = self.finger.w_vel
            paras[self.finger.index["variance_weight"]] = self.finger.w_var

            distance = []
            movement_time = []
            mean_distance_data = row[4]
            SD_distance_data = row[5]
            mean_time_data = row[6]
            SD_time_data = row[7]

            for n in range(20):
                self.finger.reset_mpcc(rx, ry, rz)
                d = (e[0] ** 2 + e[1] ** 2) ** 0.5
                w = row[2]
                self.finger.calculate_and_set_desired_velocity(d, w)

                for i in count():
                    xinit = self.finger.step(paras, recalc=True)
                    if (xinit[2] < 0 and xinit[-2] >= self.finger.theta[-1]) or i > (
                            mean_time_data + 5 * SD_time_data) / self.finger.dt:
                        d_error = ((e[0] - xinit[0]) ** 2 + (e[1] - xinit[1]) ** 2) ** 0.5
                        distance.append(d_error)
                        movement_time.append(i * self.finger.dt)
                        break

            mean_distance_simulation = np.mean(distance)
            SD_distance_simulation = np.std(distance)
            mean_time_simulation = np.mean(movement_time)
            SD_time_simulation = np.std(movement_time)

            e_mu_d = (mean_distance_simulation - mean_distance_data) ** 2
            e_sigma_d = (SD_distance_simulation - SD_distance_data) ** 2
            e_mu_mt = (mean_time_simulation - mean_time_data) ** 2
            e_sigma_mt = (SD_time_simulation - SD_time_data) ** 2

            error = e_mu_d + e_sigma_d + e_mu_mt + e_sigma_mt
            total_error += error

        print(total_error)
        return -1 * total_error

    def solve(self, init_points=2, n_iter=1000):
        logger = JSONLogger(path="ABC_Results/user_parameters_p{}.json".format(self.participant))
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        print(self.optimizer.max)

    def print_max(self):
        optimizer = BayesianOptimization(
            f=self.cost,
            pbounds=self.pbounds,
            random_state=1,
            verbose=2,
        )
        load_logs(optimizer, logs=["ABC_Results/user_parameters_p{}.json".format(self.participant)])
        x_obs = optimizer.space._params
        y_obs = optimizer.space._target
        gp = optimizer._gp
        gp.fit(x_obs, y_obs)
        print(optimizer.max)

    def plot(self, p, i):
        optimizer = BayesianOptimization(
            f=self.cost,
            pbounds=self.pbounds,
            random_state=1,
            verbose=2,
        )
        load_logs(optimizer, logs=["ABC_Results/user_parameters_p{}.json".format(self.participant)])

        x_obs = np.array([[res["params"][p]] for res in optimizer.res])[:i + 1]
        y_obs = np.array([res["target"] for res in optimizer.res])[:i + 1]
        gp = optimizer._gp
        gp.fit(x_obs, y_obs)

        # # kernel = Matern(nu=4.5)
        # kernel = RBF(length_scale=1e-5)
        # kernel = RationalQuadratic(length_scale=1,
        #                            alpha=10)  # length_scale_bounds=(1e-05, 100000.0), alpha_bounds=(1e-05, 100000.0)),
        # # kernel=DotProduct(sigma_0=3e7)
        # sigma = 1e0
        # # kernel = DotProduct(sigma_0=sigma) * DotProduct(sigma_0=sigma)
        #
        # gp = GaussianProcessRegressor(
        #     kernel=kernel,
        #     normalize_y=True,
        #     alpha=0.1,
        #     n_restarts_optimizer=25,
        # )

        # gp.fit(x_obs, y_obs)

        # max_x = np.array([optimizer.max['params'][p]])
        # max_y = optimizer.max['target']

        xmin, xmax = self.pbounds[p]
        x = np.linspace(xmin, xmax, 100)
        num_vals = len(x)
        w_vel = np.linspace(self.pbounds[p][0], self.pbounds[p][1], num_vals)
        mu, sigma = gp.predict(w_vel.reshape(-1, 1), return_std=True)

        fig = plt.figure(i)
        ax = fig.add_subplot(111)
        ax.scatter(x_obs.flatten(), y_obs, color='g', label="observations")
        # ax.scatter(max_x, max_y, color='r', label="Max value")
        ax.plot(w_vel, mu, 'k--', label="prediction")
        ax.fill_between(w_vel, mu - sigma, mu + sigma, label="SD Confidence", alpha=0.5)
        ax.set_xlabel(p)
        ax.set_ylabel("target")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-15, 1)
        plt.legend()
        counter = str(i).zfill(3)
        plt.savefig("ABC_Results\images\w_vel_p999_{}.png".format(counter))
        plt.close()
        # plt.show()

    def plot2(self):
        optimizer = BayesianOptimization(
            f=self.cost,
            pbounds=self.pbounds,
            random_state=1,
            verbose=2,
        )
        load_logs(optimizer, logs=["ABC_Results/user_parameters_p{}.json".format(self.participant)])

        x_obs = optimizer.space._params
        y_obs = optimizer.space._target
        gp = optimizer._gp

        # kernel = Matern(nu=4.5)
        kernel = RBF(length_scale=1e-5)
        kernel = RationalQuadratic(length_scale=1,
                                   alpha=10)  # length_scale_bounds=(1e-05, 100000.0), alpha_bounds=(1e-05, 100000.0)),
        # kernel=DotProduct(sigma_0=3e7)
        # sigma = 1e1
        # kernel = DotProduct(sigma_0=sigma) * DotProduct(sigma_0=sigma)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=0.5,
            n_restarts_optimizer=25,
        )

        gp.fit(x_obs, y_obs)

        max_x = np.array([optimizer.max['params']['max_motor_units'], optimizer.max['params']['w_vel']])
        max_y = optimizer.max['target']

        num_vals = 101
        heatmap = np.zeros((num_vals, num_vals))
        w_vel = np.linspace(self.pbounds["w_vel"][0], self.pbounds["w_vel"][1], num_vals)
        max_motor_units = np.linspace(self.pbounds["max_motor_units"][0], self.pbounds["max_motor_units"][1], num_vals)

        # calculate heat map
        for i in range(num_vals):
            for j in range(num_vals):
                heatmap[i, j] = gp.predict(np.array([max_motor_units[j], w_vel[i]]).reshape(1, -1))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_obs[:, 0], x_obs[:, 1], y_obs, color='g')
        ax.scatter(max_x[0], max_x[1], max_y, color='r')
        xv, yv = np.meshgrid(max_motor_units, w_vel)
        ax.plot_wireframe(xv, yv, heatmap)
        # ax.plot_surface(xv, yv, heatmap)
        ax.set_xlabel("motor")
        ax.set_ylabel("w_vel")
        plt.show()


if __name__ == '__main__':
    user_parameters = UserParameters(999)
    # user_parameters.solve()
    user_parameters.print_max()
    # for i in count():
    #     c = i + 1
    #     try:
    #         print(i)
    #         user_parameters.plot("w_vel", c)
    #     except IndexError:
    #         break
