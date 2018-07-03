import torch
import pickle
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
from cannon.utils import cuda_move


class PianoRollData(Dataset):
    def __init__(self, fname, key='train') -> None:
        self.fname = fname
        self.key = key
        self.raw_list = None
        self.parsed_song_X = None
        self.parsed_song_y = None
        self.load_data(fname, key)
        self.create_data()

    def __getitem__(self, i):
        return self.parsed_song_X[i], self.parsed_song_y[i]

    def __len__(self):
        return len(self.parsed_song_X)

    def load_data(self, fname, key):
        with open(fname, 'rb') as f:
            raw_data = pickle.load(f)
            self.raw_list = raw_data[key]

    def create_data(self):
        if self.parsed_song_X is not None:
            return
        parsed_song_X = []
        parsed_song_y = []
        for song in self.raw_list:
            X_song = torch.zeros(len(song), 88)
            # sparse to dense conversion
            for t, t_list in enumerate(song):
                for note in t_list:
                    note = note - 21  # piano roll notes have range [21, 108]
                    X_song[t, note] = 1

            parsed_song_X.append(torch.tensor(X_song[:-1]))
            parsed_song_y.append(torch.tensor(X_song[1:]))
        self.parsed_song_X = parsed_song_X
        self.parsed_song_y = parsed_song_y

    def get_one_hot_list(self):
        """ Deprecated!
            Iterator over the whole dataset with batch size=1.
            Pass the entire sequence and does not need masking.
        """
        if not self.parsed_song_X:
            self.create_data()

        for x, y in zip(self.parsed_song_X, self.parsed_song_y):
            yield Variable(x.unsqueeze(1)), Variable(y.unsqueeze(1))

    @staticmethod
    def frame_level_accuracy(y_out, target):
        assert len(y_out.size()) == 2
        y_out = (y_out > .5).float()
        TP = torch.sum(target * y_out, dim=1)
        FP = torch.sum((1 - target) * y_out, dim=1)
        FN = torch.sum(target * (1 - y_out), dim=1)

        den = (TP + FP + FN)
        TP = TP[den > 0]
        den = den[den > 0]

        acc = TP / den
        return torch.mean(acc)

    @staticmethod
    def compute_metrics_packed(model, data, batch_size=128):
        X_train, y_train = [], []
        for x, y in data.get_one_hot_list():
            X_train.append(x)
            y_train.append(y)

        me = 0.0
        acc = 0.0
        n_batch = 0
        sum_masks = 0
        with torch.no_grad():
            for i in range(0, len(X_train), batch_size):
                max_idx = min(batch_size, len(X_train) - i)
                max_len = max([X_train[i + k].size(0) for k in range(max_idx)])
                X_i = Variable(torch.zeros(max_len, max_idx, 88))
                y_i = Variable(torch.zeros(max_len, max_idx, 88))
                mask_i = Variable(torch.zeros(max_len, max_idx, 1))
                X_i, y_i, mask_i = cuda_move(X_i), cuda_move(y_i), cuda_move(mask_i)

                for k in range(max_idx):
                    l = X_train[i + k].size(0)
                    X_i[:l, k:k + 1, :] = X_train[i + k]
                    y_i[:l, k:k + 1, :] = y_train[i + k]
                    mask_i[:l, k:k + 1, :] = 1

                y_out = model.forward(X_i)
                y_out = y_out * mask_i.expand_as(y_out)
                y_i = y_i * mask_i.expand_as(y_i)

                me += F.binary_cross_entropy_with_logits(y_out, y_i).item()
                y_out = F.sigmoid(y_out)

                for j in range(y_out.size(1)):
                    curr_acc = PianoRollData.frame_level_accuracy(y_out[:, j, :], y_i[:, j, :]).item()
                    acc += X_train[i + j].size(0) * curr_acc
                    sum_masks += X_train[i + j].size(0)

                n_batch += max_idx
        return me / n_batch, acc / sum_masks

    def __str__(self):
        return "Data from {}. (key={})".format(self.fname, self.key)
