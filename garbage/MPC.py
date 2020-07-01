import osqp
import numpy as np
from scipy import sparse
from itertools import count
from utils.MPC_plotter import Plotter

import torch

torch.from_numpy()

def step_sample(desired_force):
    max_motor_units = 10
    average_activation = 0.1
    a = desired_force / (max_motor_units * average_activation)
    print(a)
    return np.random.normal(max_motor_units * a,
                            np.sqrt(max_motor_units * abs(a) * abs(1 - abs(a)))) * average_activation


dt = 1e-3  # seconds
dt2 = dt ** 2
dt3 = dt ** 3
m = 0.05  # kilograms
Ad = sparse.csc_matrix([
    [1., 0., 0., dt, 0., 0., dt2 / m, 0., 0.],  # x
    [0., 1., 0., 0., dt, 0., 0., dt2 / m, 0.],  # y
    [0., 0., 1., 0., 0., dt, 0., 0., dt2 / m],  # z
    [0., 0., 0., 1., 0., 0., dt / m, 0., 0.],  # dx
    [0., 0., 0., 0., 1., 0., 0., dt / m, 0.],  # dy
    [0., 0., 0., 0., 0., 1., 0., 0., dt / m],  # dz
    [0., 0., 0., 0., 0., 0., 1., 0., 0.],  # fx
    [0., 0., 0., 0., 0., 0., 0., 1., 0.],  # fy
    [0., 0., 0., 0., 0., 0., 0., 0., 1.],  # fz

])
Bd = sparse.csc_matrix([
    [dt3 / m, 0, 0],
    [0, dt3 / m, 0],
    [0, 0, dt3 / m],
    [dt2 / m, 0, 0],
    [0, dt2 / m, 0],
    [0, 0, dt2 / m],
    [dt, 0, 0],
    [0, dt, 0],
    [0, 0, dt]
])
[nx, nu] = Bd.shape

# Constraints
umin = np.array([-50, -50, -50])
umax = np.array([50, 50, 50])
xmin = np.array([-0.05, -0.05, -0.05, -1, -1, -1, -1, -1, -1])
xmax = np.array([0.05, 0.05, 0.05, 1, 1, 1, 1, 1, 1])

# Objective function
Q = sparse.diags([1., 1., 1., 0.0001, 0.0001, 0.0001, 0, 0, 0])
QN = Q
R = 0.000001*sparse.eye(nu)

# Initial and reference states
x0 = np.zeros(nx)
xr = np.array([-0.01, 0.02, 0.03, 0., 0., 0., 0., 0., 0.])

# Prediction horizon
N = 500

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N * nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N * nx)])
ueq = leq

# - input and state constraints
Aineq = sparse.eye((N + 1) * nx + N * nu)
lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])

# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

plotter = Plotter()
for i in count():
    if i % 200 == 0:
        xr *= -1
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(N * nu)])
        prob.update(q=q)

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N * nu:-(N - 1) * nu]
    x0[6] = step_sample(x0[6])
    x0[7] = step_sample(x0[7])
    x0[8] = step_sample(x0[8])

    x0 = Ad.dot(x0) + Bd.dot(ctrl)
    # # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

    plotter.update(i,
                   [xr[0], xr[1], xr[2]], [xr[3], xr[4], xr[5]], [xr[6], xr[7], xr[8]],
                   [x0[0], x0[1], x0[2]], [x0[3], x0[4], x0[5]], [x0[6], x0[7], x0[8]])
    plotter.redraw()
