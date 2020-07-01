from forces_pro import *
import FORCESNLPsolver_py
import numpy as np

if __name__ == '__main__':
    problem = FORCESNLPsolver_py.FORCESNLPsolver_params
    model = {}
    model['nvar'] = 12
    model["N"] = 100
    x_planned = np.zeros([int(model['nvar']), int(model['N'])])
    temp_x0 = x_planned.reshape(int(model['nvar'] * model['N']), order='F')
    problem['x0'] = temp_x0

    xinit = x_planned[:9,-1]
    parameters = np.zeros((5,100))
    parameters[0:3,:] =  1e-2
    parameters[0, :] = 1
    parameters[0, :] = 1e-2
    problem['all_parameters'] = np.asarray(parameters.reshape((5*100), order='F'))
    problem['xinit'] = np.asarray(xinit)
    [solverout, exitflag, info] = FORCESNLPsolver_py.FORCESNLPsolver_solve(problem)