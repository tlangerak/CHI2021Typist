from mpcc_forces import FingerController
import numpy as np
from itertools import count
from planner import Planner
import csv

class DataCreator:
    def __init__(self, participant):
        self.finger = FingerController()
        self.finger.setup_mpcc(False)
        self.planner = Planner()
        self.participant = participant
        filen = "fittslaw_simple_p{}.csv".format(participant)
        my_data = np.genfromtxt(filen, delimiter=',')
        self.my_data = my_data[2:]
        self.pixels_per_meter = (900 / 100) * 1e3
        self.saved_data =[]

    def create(self):
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
                100,
                0.01, 0.01, 0.01
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

            for n in range(100):
                self.finger.reset_mpcc(rx, ry, rz)
                d = (e[0] ** 2 + e[1] ** 2) ** 0.5
                w = row[2]
                self.finger.calculate_and_set_desired_velocity(d, w)

                mean_time_data = row[4]
                SD_time_data = row[5]

                for i in count():
                    xinit = self.finger.step(paras, recalc=True)
                    if (xinit[2] < 0 and xinit[-2] >= self.finger.theta[-1]):
                        d = ((e[0] - xinit[0]) ** 2 + (e[1] - xinit[1]) ** 2) ** 0.5
                        distance.append(d)
                        movement_time.append(i * self.finger.dt)
                        break

            mean_distance_simulation = np.mean(distance)
            SD_distance_simulation = np.std(distance)
            mean_time_simulation = np.mean(movement_time)
            SD_time_simulation = np.std(movement_time)
            data = [row[0], row[1], row[2], row[3], mean_distance_simulation, SD_distance_simulation, mean_time_simulation, SD_time_simulation]
            self.saved_data.append(data)

    def save(self):
        header = ["x", "y", "w", "h", "mu_e", "sigma_e", "mu_mt", "sigma_mt"]
        with open('fittslaw_simple_p999.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(self.saved_data)

if __name__ == '__main__':
    synth = DataCreator(0)
    # synth.create()
    synth.save()