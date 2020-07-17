"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gym
import numpy as np
import torch
import logging
import math
from pytorch_mppi import mppi_finger as mppi
from gym import wrappers, logger as gym_log
import random
import custom_envs
from torch.utils.tensorboard import SummaryWriter

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

writer = SummaryWriter()

if __name__ == "__main__":
    ENV_NAME = "Finger-v1"
    TIMESTEPS = 50  # T >> Horizon
    N_SAMPLES = 250  # K >> number of parallel samples
    ACTION_LOW = 0.
    ACTION_HIGH = 1.

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    nx = 18
    nu = 7

    noise_sigma = torch.tensor(0.0001 * torch.eye(nu).clone().detach(), device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.

    randseed = None
    if randseed is None:
        randseed = random.randint(0, 1000000)

    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 100
    BOOT_STRAP_ITER = 500

    dataset = None

    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, H_UNITS*2),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS*2, H_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=d)


    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        xu = torch.cat((state, u), dim=1)
        state_residual = network(xu)
        next_state = state + state_residual
        return next_state


    def running_cost(state, action):
        ex = state[:, -1] ** 2 + state[:, -2] ** 2 + state[:, -3] ** 2
        eu = torch.sum(torch.mm(action, action.T), dim=1)
        cost = ex + 0.00001 * eu
        return cost


    def train(new_data):
        global dataset
        # not normalized inside the simulator
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -3:] = torch.clamp(new_data[:, -3:], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)

        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        indices = torch.tensor(random.sample(range(dataset.shape[0]), min(dataset.shape[0], 5000)))
        bootstrapped_data = dataset[indices].clone()

        xu = bootstrapped_data[:-1]
        Y = bootstrapped_data[1:, :nx] - bootstrapped_data[:-1, :nx]

        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(xu)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False


    env = gym.envs.make(ENV_NAME)
    env._max_episode_steps = 500
    env.reset()

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=nu)
            env.step(action)
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        logger.info("bootstrapping finished")

    # env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))

    logger.info("Starting Runs")
    total_reward, data = mppi.run_mppi(mppi_gym, env, train, retrain_after_iter=499, render=False, writer=writer)
    logger.info("Total reward %f", total_reward)
