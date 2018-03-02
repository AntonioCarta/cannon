import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple


class CopyDataset:
    def __init__(self, n_timesteps: int, n_features: int, cuda=False) -> None:
        self.t_steps = n_timesteps
        self.n_feat = n_features
        self.cuda = cuda
        self.curr_i = -1
        self.temp_data = None

    def get_batch(self, size: int=32, pad=True) -> Tuple[Variable, Variable]:
        data_dim = 4096
        if (self.temp_data is None) or (self.curr_i + size >= self.temp_data[0].size(1)):
            a = np.random.randint(0, 2, size=(self.t_steps, data_dim, self.n_feat))
            if pad:
                X = Variable(torch.zeros((2 * self.t_steps + 2, data_dim, self.n_feat + 1)))
                y = Variable(torch.zeros((2 * self.t_steps + 2, data_dim, self.n_feat + 1)))
                ta = torch.from_numpy(a)

                X[:self.t_steps, :, :-1] = ta
                X[self.t_steps, :, -1] = 1.0

                y[self.t_steps + 1:-1, :, :-1] = ta
                y[-1, :, -1] = 1.0
            else:
                X = a
                y = X

            if self.cuda:
                X = X.cuda()
                y = y.cuda()
            self.temp_data = X, y
            self.curr_i = 0

        i = self.curr_i
        X = self.temp_data[0][:, i:i + size, :]
        y = self.temp_data[1][:, i:i + size, :]
        self.curr_i = i + size
        return X, y
