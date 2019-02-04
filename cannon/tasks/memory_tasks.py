import torch
import torch.nn.functional as F
import numpy as np


class CopyTask:
    def __init__(self, S, T, K=2):
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
        """
        self.S = S
        self.T = T
        self.K = K

    def get_batch(self, batch_size):
        rand_seq = torch.randint(0, self.K, (self.S, batch_size)).long()
        blank_T = self.K * torch.ones((self.T, batch_size)).long()
        blank_S = self.K * torch.ones((self.S, batch_size)).long()

        input_seq = torch.cat([rand_seq, blank_T, blank_S], dim=0)
        input_seq[self.T + self.S - 1, :] = self.K + 1
        output_seq = torch.cat([blank_S, blank_T, rand_seq], dim=0)
        return input_seq, output_seq

    def score(self, y_pred, y_target):
        assert y_pred.shape[-1] == self.K + 2
        return F.cross_entropy(y_pred.reshape(-1, self.K + 2), y_target.reshape(-1))


class AdditionTask:
    def __init__(self, seq_len):
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

    def score(self, y_pred, y_target):
        return F.mse_loss(y_pred, y_target)


def test_copy():
    S, T, K = 10, 5, 2
    batch_size = 3
    data = CopyTask(S, T, K)
    x, y = data.get_batch(batch_size)

    # marker at correct position
    assert (x[T + S - 1] == K + 1).all()

    mask = torch.cat([y[:T + S], x[:S]], dim=0)
    out = torch.zeros((T + 2*S, batch_size, K + 2))

    # optimal solution has zero error (logits should be +inf).
    for t in range(2*S + T):
        for b in range(batch_size):
            out[t, b, mask[t, b]] = 10**9
    loss = data.score(out, y)
    assert loss == 0.0


def test_addition():
    seq_len = 10
    batch_size = 4

    data = AdditionTask(seq_len)
    x, y = data.get_batch(batch_size)

    # two and only two positions are marked for each sequence
    for b in range(batch_size):
        assert x[:, b, 1].sum() == 2.0

    # optimal solution has zero error
    out = (x[:, :, 0] * x[:, :, 1]).sum(dim=0).reshape(batch_size, 1)
    assert data.score(out, y) == 0.0


if __name__ == '__main__':
    print("main.")
    test_copy()
    test_addition()

