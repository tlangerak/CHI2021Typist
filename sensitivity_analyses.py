from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from find_user_parameters import UserParameters
import json

if __name__ == '__main__':
    user = UserParameters(999)
    problem = {
        'num_vars': 13,
        'names': ['average_activation', 'max_motor_units', 'w_contour', 'w_input_t', 'w_input_xy',
                  'w_input_z', 'w_lag', 'w_vel', 'x_target', 'y_target', "w_target", "a", "b"],
        'bounds': [[1e-1, 1e0],  # activation
                   [1e2, 1e4],  # units
                   [1e1, 1e3],  # contour
                   [1e-6, 1e0],  # in
                   [1e-6, 1e0],  # in
                   [1e-6, 1e0],  # in
                   [1e1, 1e4],  # lag
                   [1e0, 1e3],  # vel
                   [0, 2.5e-2],  # x
                   [0, 2.5e-2],  # y
                   [1e-3, 1e-2],  # w
                   [3e-1, 6e-1],  # a
                   [3e-1, 6e-1]  # b
                   ]

    }
    assert problem['num_vars'] == len(problem['names']) and problem['num_vars'] == len(problem['bounds'])

    param_values = np.array(saltelli.sample(problem, 1))
    Y = np.zeros([param_values.shape[0]])
    print(param_values.shape)
    for i, X in enumerate(param_values):
        print(i)
        d = np.sqrt(X[8] ** 2 + X[9] ** 2)
        mu_mt = X[11] + X[12] + np.log2(d / X[10] + 1)
        sd_mt = 0.48 * mu_mt
        target = [X[8], X[9], X[10], X[10], 0, 0, mu_mt, sd_mt]
        Y[i] = user.evaluate(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], target)

    Si = sobol.analyze(problem, Y)
    with open('Sensitivity_Analyses.json', 'w') as fp:
        json.dump(Si, fp)

    print(Si['S1'])
    print(Si['ST'])
    print("x1-x2:", Si['S2'][0, 1])
    print("x1-x3:", Si['S2'][0, 2])
    print("x2-x3:", Si['S2'][1, 2])
