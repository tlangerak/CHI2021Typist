from forces_pro import *
import FORCESNLPsolver_py
import numpy as np

if __name__ == '__main__':
    # Define solver options
    codeoptions = forcespro.CodeOptions()
    codeoptions.maxit = 200  # Maximum number of iterations
    codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but not for timings)
    codeoptions.optlevel = 3  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
    codeoptions.nlp.integrator.Ts = integrator_stepsize
    codeoptions.nlp.integrator.nodes = 5
    codeoptions.nlp.integrator.type = 'ERK4'
    codeoptions.solvemethod = 'SQP_NLP'
    codeoptions.sqp_nlp.rti = 1
    codeoptions.sqp_nlp.maxSQPit = 1

    # Generate real-time SQP solver
    solver = model.generate_solver(codeoptions)