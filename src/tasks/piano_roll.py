import torch
import pickle
from typing import Tuple
from torch.autograd import Variable


class PianoRollData:
    def __init__(self, fname, key='train', timesteps=50, skip=10) -> None:
        self.fname = fname
        self.key = key
        self.timesteps = timesteps
        self.skip = skip
        self.raw_list = None
        self.X = None
        self.y = None
        self.parsed_song_X = None
        self.parsed_song_y = None
        self.masks = None
        self.load_data(fname, key)
        self.create_data()

    def __getitem__(self, i):
        if self.X is None:
            self.get_data()
        return self.X[:, i, :].data, self.y[:, i, :].data

    def __len__(self):
        if self.X is None:
            self.get_data()
        return self.X.size(1)

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

            parsed_song_X.append(X_song[:-1])
            parsed_song_y.append(X_song[1:])
        self.parsed_song_X = parsed_song_X
        self.parsed_song_y = parsed_song_y

    def get_data(self) -> Tuple[Variable, Variable, Variable]:
        if self.X is not None and self.y is not None:
            # Data has been already created
            return self.X, self.y, self.masks

        if self.parsed_song_X is None:
            self.create_data()

        batches_X = []
        batches_y = []
        masks = []
        a = torch.arange(0, self.timesteps)
        for X_song in self.parsed_song_X:
            # create batch from song tensor
            min_batch_size = self.timesteps // 4
            for i in range(0, X_song.size(0) - min_batch_size - 1, self.skip):
                t_n = min(X_song.size(0) - i, self.timesteps + 1)
                x_song_n = torch.zeros(self.timesteps, 88)
                y_song_n = torch.zeros(self.timesteps, 88)
                x_song_n[:t_n-1, :] = X_song[i:i + t_n - 1]
                y_song_n[:t_n-1, :] = X_song[i + 1:i + t_n]
                mask_n = a < t_n - 1
                batches_X.append(x_song_n)
                batches_y.append(y_song_n)
                masks.append(mask_n)

        # dim: (batch, time, feat)
        self.X = torch.stack(batches_X)
        self.y = torch.stack(batches_y)
        self.masks = torch.stack(masks)
        self.X = Variable(torch.transpose(self.X, 0, 1))
        self.y = Variable(torch.transpose(self.y, 0, 1))
        self.masks = Variable(torch.transpose(self.masks, 0, 1).float().unsqueeze(2))
        return self.X, self.y, self.masks

    def get_one_hot_list(self):
        """ Iterator over the whole dataset with batch size=1. Pass the entire sequence and does not need masking. """
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

    def __str__(self):
        return "Data from {}. (key={}, t_step={}, skip={})".format(self.fname, self.key, self.timesteps, self.skip)
