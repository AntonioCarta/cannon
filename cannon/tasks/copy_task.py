import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple
from .interface import Dataset


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


class WarpedCopyTask(Dataset):
    def __init__(self, S, T, K=2):
        """
        Warped Copy task.
        The model must be able to memorize a sequence of S elements emitted in random positions.
        Create a sequence of T + 1 + S elements from an alphabet of size K.
        Each input sequence consists of:
        - S characters are emitted in random positions [1, T], uniformly sampled from {a_1, ..., a_k}
        - the other characters in the first T positions are set to a_{k+1}
        - position T+1 is set to a_{k+2}  (the output delimiter)
        - the next S elements are set to a_{k+1}

        The expected output consists of:
        - T + 1 elements set to a_{k+1}
        - S elements corresponding to the S input elements from {a_1, ..., a_k}

        the performance of the model is evaluated with the crossentropy error over the entire sequence.

        Args:
            S (int): sequence length
            T (int): blank length
            K (int): alphabet size
        """
        super().__init__()
        self.S = S
        self.T = T
        self.K = K

    @property
    def input_shape(self):
        return (self.T + 1 + self.S, 1)

    @property
    def output_shape(self):
        return (self.T + 1 + self.S, 1, self.K + 2)

    def get_batch(self, batch_size):
        rand_seq = torch.randint(0, self.K, (self.S, batch_size)).long()
        blank_Sp1 = self.K * torch.ones((self.S + 1, batch_size)).long()
        blank_Tp1 = self.K * torch.ones((self.T + 1, batch_size)).long()
        first_T = self.K * torch.ones((self.T, batch_size)).long()

        for bi in range(batch_size):
            pos = np.random.choice(self.T, self.S, replace=False)
            pos = np.sort(pos)
            pos = torch.tensor(pos)
            first_T[pos,bi] = rand_seq[:, bi]

        input_seq = torch.cat([first_T, blank_Sp1], dim=0)
        input_seq[self.T, :] = self.K + 1

        output_seq = torch.cat([blank_Tp1, rand_seq], dim=0)
        return cuda_move(input_seq), cuda_move(output_seq)

    def loss_score(self, y_pred, y_target):
        assert y_pred.shape[-1] == self.K + 2
        return F.cross_entropy(y_pred.reshape(-1, self.K + 2), y_target.reshape(-1))

    def metric_score(self, y_pred, y_target):
        return (y_pred.argmax(dim=2) == y_target).float().mean()

    def visualize_sample(self, x, y):
        str_x = " in:" + "".join(str(el) for el in x.tolist())
        str_y = "out:" + "".join(str(el) for el in y.argmax(dim=1).tolist())
        return str_x + '\n' + str_y
