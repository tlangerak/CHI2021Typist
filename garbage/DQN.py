from garbage.SecondOrderLag import Agent
import random
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, states, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(states, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = 5
n_states = 8
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    return policy_net(state)
    # global steps_done
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #                 math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    # if sample > eps_threshold:
    #     with torch.no_grad():
    #         return policy_net(state).max(1)[1].view(1, 1), True
    # else:
    #     return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long), False


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class SingleTarget:
    def __init__(self):
        self.agent = Agent()
        self.target = [0.15, 0.3]
        self.agent.update_target(self.target)
        self.state = self.agent.state0
        self.target_x_width = 0.05
        self.target_y_width = 0.05

    def reset(self):
        self.__init__()

    def step(self, action):
        self.agent.k = action[:3]
        self.agent.d = action[3]
        self.agent.m = action[4]
        self.state, done = self.agent.step()
        agent_pos = [self.state[0], self.state[2]]
        x_error = ((agent_pos[0] - self.target[0]) ** 2) ** 0.5
        y_error = ((agent_pos[1] - self.target[1]) ** 2) ** 0.5

        if x_error < self.target_x_width / 2 and y_error < self.target_y_width / 2 and done:
            reward = 10
        else:
            reward = -1 * (x_error ** 2 + y_error ** 2)

        return self.state, reward, done

    def get_state(self):
        return torch.cat((torch.tensor(self.state).float(), torch.tensor(self.target).float()), 0)


num_episodes = 50
env = SingleTarget()

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = env.get_state()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done = env.step(action)
        reward = torch.tensor([reward])
        if not done:
            next_state = env.get_state()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
