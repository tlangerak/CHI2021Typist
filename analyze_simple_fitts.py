import json
import numpy as np
from itertools import count
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fitts_law_shannon(x, a, b):
    MT = a + b * x
    return MT


def ID(d, w):
    return np.log2((d / w) + 1)


if __name__ == '__main__':
    filen = "fittslaw_simple_p0.json"
    with open(filen, 'r') as json_file:
        data = json.load(json_file)
    print(data)

    x = []
    y = []
    for i in count():
        idx = "repetition_{}".format(i)
        try:
            if data[idx]["target_x"] != 960 and data[idx]["target_y"] != 600:
                # if data[idx]["hit"]:
                    d = (((data[idx]['start_x'] - data[idx]['target_x']) ** 2 +
                          (data[idx]['start_y'] - data[idx]['target_y']) ** 2) ** 0.5)/58
                    w = data[idx]['target_w']/58
                    if w != 0:
                        x.append(ID(d, w))
                        y.append(data[idx]['movement_time'] * 1e-3)
        except KeyError:
            print("Quit on index: {}".format(i))
            break

    popt, pcov = curve_fit(fitts_law_shannon, x, y)
    print("PARAMERS: {}".format(popt))
    plt.scatter(x, y)
    x.sort()
    plt.plot(x, fitts_law_shannon(np.asarray(x), *popt), "b--")
    plt.show()
