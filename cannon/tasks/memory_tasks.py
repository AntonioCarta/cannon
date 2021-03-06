import torch
import torch.nn.functional as F
import numpy as np
from cannon.utils import cuda_move


class CopyTask:
    def __init__(self, S, T, K=2, batch_size=64):
        """
        Copy task.
        The model must be able to memorize a sequence of S elements and reconstruct it after T timesteps.
        Create a sequence of 2S + T elements from an alphabet of size K.
        Each input sequence consists of:
        - S elements randomly sampled from {a_1, ..., a_k}
        - T-1 elements set to a_{k+1}  (a blank element)
        - 1 element set to a_{k+2}  (the output delimiter)
        - S elements set to a_{k+1}

        The expected output consists of:
        - T + S elements set to a_{k+1}
        - S elements corresponding to the first S input elements

        the performance of the model is evaluated with the crossentropy error over the entire sequence.

        Args:
            S (int): sequence length
            T (int): blank length
            K (int): alphabet size
            batch_size (int): batch size
        """
        self.S = S
        self.T = T
        self.K = K
        self.batch_size = batch_size

    @property
    def input_shape(self):
        """ Input shape. If batch is not set by the class, it is set to 1. """
        return 2 * self.S + self.T, 1

    @property
    def output_shape(self):
        """ Output shape. If batch is not set by the class, it is set to 1. """
        return 2 * self.S + self.T, 1, self.K + 2

    @property
    def input_size(self):
        return 1

    @property
    def output_size(self):
        return self.K + 2

    def get_batch(self, batch_size):
        rand_seq = torch.randint(0, self.K, (self.S, batch_size, 1)).long()
        blank_T = self.K * torch.ones((self.T, batch_size, 1)).long()
        blank_S = self.K * torch.ones((self.S, batch_size, 1)).long()

        input_seq = torch.cat([rand_seq, blank_T, blank_S], dim=0)
        input_seq[self.T + self.S - 1, :] = self.K + 1
        output_seq = torch.cat([blank_S, blank_T, rand_seq], dim=0)
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

    def iter(self):
        for ii in range(1000):
            yield self.get_batch(self.batch_size)


class AdditionTask:
    def __init__(self, seq_len):
        """
        Dataset for the adding problem.
        Each sequence element consists of two features:
        - a number uniformly sampled from [0, 1)
        - a binary marker. Exactly two positions (randomly sampled) are set to 1, while the others are set to 0
        The output is a scalar value corresponding to the sum of the two elements with marker value equal to 1.

        Args:
            seq_len: length of the sequences
        """
        self.seq_len = seq_len

    def get_batch(self, batch_size):
        rand_nums = torch.rand(self.seq_len, batch_size)
        rand_markers = torch.zeros(self.seq_len, batch_size)

        idxs = np.arange(0, self.seq_len)
        for b in range(batch_size):
            np.random.shuffle(idxs)
            rand_markers[idxs[0], b] = 1.0
            rand_markers[idxs[1], b] = 1.0

        x = torch.stack([rand_nums, rand_markers], dim=2)
        y = torch.sum(rand_nums * rand_markers, dim=0).reshape(batch_size, 1)
        return x, y

    def loss_score(self, y_pred, y_target):
        return F.mse_loss(y_pred, y_target)
