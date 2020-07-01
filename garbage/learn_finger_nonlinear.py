import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from barbar import Bar

class Learner:
    def __init__(self):
        print("START DATA LOADING")
        self.batch_size = 1
        self.data_params = {'batch_size': self.batch_size,
                            'shuffle': True,
                            'num_workers': 6}
        df = pd.read_csv("data_finger_v0.csv")
        df['split'] = np.random.randn(df.shape[0], 1)
        msk = np.random.rand(len(df)) <= 0.7
        self.train_data = FingerDataset(df[msk])
        self.test_data = FingerDataset(df[~msk])
        self.training_generator = torch.utils.data.DataLoader(self.train_data, **self.data_params)
        self.test_generator = torch.utils.data.DataLoader(self.test_data, **self.data_params)
        print("DATA LOADING DONE")

        self.loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-4
        self.n_epochs = 1000
        n_hidden = 256
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.train_data.n_actions + self.train_data.n_observation, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, self.train_data.n_target),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()

    def train(self):
        self.model.train()
        for e in range(self.n_epochs):
            print("Epoch: {} of {}".format(e, self.n_epochs))
            total_loss = 0
            for x, y in Bar(self.training_generator):
                self.model.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().numpy())

            self.writer.add_scalar('Loss/train', total_loss, e)
            if e % 10 == 0:
                self.test(e)

    def test(self, e=None):
        self.model.eval()
        total_loss = 0
        for x, y in Bar(self.test_generator):
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            total_loss += float(loss.detach().numpy())
        self.writer.add_scalar('Loss/test', total_loss, e)


class FingerDataset(Dataset):
    def __init__(self, data):
        self.n_observation = 9
        self.n_actions = 7
        self.n_target = 9
        self.data = data
        header = []
        for n in range(7):
            header.append("action_{}".format(n))
        header.append("init_x")
        header.append("init_y")
        header.append("init_z")
        header.append("init_dx")
        header.append("init_dy")
        header.append("init_dz")
        header.append("init_ddx")
        header.append("init_ddy")
        header.append("init_ddz")
        header.append("result_x")
        header.append("result_y")
        header.append("result_z")
        header.append("result_dx")
        header.append("result_dy")
        header.append("result_dz")
        header.append("result_ddx")
        header.append("result_ddy")
        header.append("result_ddz")
        self.header = header

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x=[]
        for h in self.header[:self.n_actions+self.n_observation]:
            x.append(self.data.iloc[idx, :][h])

        y = []
        for h in self.header[self.n_actions+self.n_observation:]:
            y.append(self.data.iloc[idx, :][h])

        sample = [torch.tensor(x).float(), torch.tensor(y).float()]
        return sample


if __name__ == '__main__':
    l = Learner()
    l.train()
