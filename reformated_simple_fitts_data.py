# 1. Load JSON
# 2. Collect all reps with same target (x,y,w,h)
# 3. calculate mean and std on error on time
# 4. save in csv (x,y,w,h, mu_d, sigma_d, mu_mt, sigma_mt)

from itertools import count
import json
import csv
import numpy as np

filen = "fittslaw_simple_p0.json"


def check_if_datapoint_exists(new_point, existing_points):
    if len(existing_points) > 0:
        for i, l in enumerate(existing_points):
            if l[:4] == new_point[:4]:
                return True, i
    return False, None


if __name__ == '__main__':
    saved_data = []
    with open(filen, 'r') as json_file:
        data = json.load(json_file)
        for i in count():
            idx = "repetition_{}".format(i)
            try:
                if data[idx]["target_x"] != 960 and data[idx]["target_y"] != 600:
                    # check if we have this combination already
                    # if we dont create if
                    error = (((data[idx]['end_x'] - data[idx]['target_x']) ** 2 +
                              (data[idx]['end_y'] - data[idx]['target_y']) ** 2) ** 0.5)
                    p_per_m = (900/100)*1e3
                    new_data = [data[idx]["target_x"]/p_per_m-960/p_per_m,
                                data[idx]["target_y"]/p_per_m-600/p_per_m,
                                data[idx]["target_w"]/p_per_m,
                                data[idx]["target_h"]/p_per_m,
                                error/p_per_m,
                                data[idx]["movement_time"]*1e-3]
                    exists, index = check_if_datapoint_exists(new_data, saved_data)
                    print(exists)
                    if exists:
                        saved_data[index][4].append(new_data[4])
                        saved_data[index][5].append(new_data[5])
                        print(saved_data[index])
                    else:
                        saved_data.append(new_data[:4])
                        saved_data[-1].append([new_data[4]])
                        saved_data[-1].append([new_data[5]])
                        pass

            except KeyError:
                print("Quit on index: {}".format(i))
                break

    for row in saved_data:
        print(row)
        row.append(np.mean(row[4]))
        row.append(np.std(row[4]))
        row.append(np.mean(row[5]))
        row.append(np.std(row[5]))
        row.pop(5)
        row.pop(4)
        print(row)

    header = ["x", "y", "w", "h", "mu_e", "sigma_e", "mu_mt", "sigma_mt"]
    with open('fittslaw_simple_p0.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(saved_data)