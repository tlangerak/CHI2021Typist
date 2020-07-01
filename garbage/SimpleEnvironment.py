from garbage.SecondOrderLag import Agent
import torch

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

        return torch.cat((torch.tensor(self.state).float(), torch.tensor(self.target).float()), 0), reward, done

    def get_state(self):
        return torch.cat((torch.tensor(self.state).float(), torch.tensor(self.target).float()), 0)