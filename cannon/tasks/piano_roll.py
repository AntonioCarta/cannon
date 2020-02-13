import torch
import pickle
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
from cannon.utils import cuda_move


class PianoRollData(Dataset):
    def __init__(self, fname, key='train', batch_size=1) -> None:
        self.fname = fname
        self.key = key
        self.batch_size = batch_size
        self.raw_list = None
        self.xs = None
        self.ys = None
        self.load_data(fname, key)
        self.parse_raw_data()
        self.loss_mean = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def __getitem__(self, i):
        return self.xs[i], self.ys[i]

    def __len__(self):
        return len(self.xs)

    def load_data(self, fname, key):
        with open(fname, 'rb') as f:
            raw_data = pickle.load(f)
            self.raw_list = raw_data[key]

    def parse_raw_data(self):
        if self.xs is not None:
            return
        parsed_song_X, parsed_song_y = [],  []
        for song in self.raw_list:
            X_song = torch.zeros(len(song), 88)
            for t, t_list in enumerate(song):  # sparse to dense conversion
                for note in t_list:
                    note = note - 21  # piano roll notes have range [21, 108]
                    X_song[t, note] = 1
            parsed_song_X.append(torch.tensor(X_song[:-1]))
            parsed_song_y.append(torch.tensor(X_song[1:]))
        self.xs = parsed_song_X
        self.ys = parsed_song_y

    def iter(self):
        n_samples = len(self.xs)
        if self.key == 'train':  # shuffle dataset
            idxs = torch.randperm(n_samples)
            self.xs = [self.xs[ii] for ii in idxs]
            self.ys = [self.ys[ii] for ii in idxs]
        for ii in range(0, len(self.xs), self.batch_size):
            x_batch = self.xs[ii:ii + self.batch_size]
            y_batch = self.ys[ii:ii + self.batch_size]
            t_batch = torch.tensor([el.shape[0] for el in x_batch])
            bs = len(x_batch)
            t_max = max(el.shape[0] for el in x_batch)
            x = torch.zeros(t_max, bs, x_batch[0].shape[-1])
            y = torch.zeros(t_max, bs, x_batch[0].shape[-1])
            for bi, (x_el, y_el) in enumerate(zip(x_batch, y_batch)):
                x[:x_el.shape[0], bi] = x_el
                y[:x_el.shape[0], bi] = y_el
            x = cuda_move(x)
            y = cuda_move(y)
            t_batch = cuda_move(t_batch)
            yield x, (y, t_batch)

    def metric_score(self, batch, y_pred):
        """ Accuracy score. """
        return -self.loss_score(batch, y_pred)

    def loss_score(self, batch, y_pred):
        assert len(y_pred.shape) == 3
        x, y_target = batch
        y_true, t_batch = y_target
        yp = []
        yt = []
        for bi in range(y_pred.shape[1]):
            yp.append(y_pred[:t_batch[bi], bi])
            yt.append(y_true[:t_batch[bi], bi])
        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)
        err = self.loss_mean(yp, yt)
        return err

    def summed_loss_score(self, batch, y_pred):
        assert len(y_pred.shape) == 3
        x, y_target = batch
        y_true, t_batch = y_target
        yp = []
        yt = []
        for bi in range(y_pred.shape[1]):
            yp.append(y_pred[:t_batch[bi], bi])
            yt.append(y_true[:t_batch[bi], bi])
        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)
        err = self.loss_sum(yp, yt)
        return err, yt.shape[0]

    @staticmethod
    def frame_level_accuracy(batch, y_pred):
        assert len(y_pred.size()) == 3
        x, y = batch
        y_true, t_batch = y

        yp = []
        yt = []
        for bi in range(y_pred.shape[1]):
            yp.append(y_pred[:t_batch[bi], bi])
            yt.append(y_true[:t_batch[bi], bi])
        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)

        y_pred = (yp > .5).float()
        TPt = torch.sum(yt * y_pred, dim=1)
        FPt = torch.sum((1 - yt) * y_pred, dim=1)
        FNt = torch.sum(yt * (1 - y_pred), dim=1)

        num = TPt
        den = (TPt + FPt + FNt)
        num[den == 0] = 0  # gestire il caso in cui den=0, settando l'accuracy a zero (perchè implica TP=0)
        den[den == 0] = 1

        num = num.sum()
        den = den.sum()
        return (num / den) * yt.shape[0], yt.shape[0]

    def __str__(self):
        return "Data from {}. (key={})".format(self.fname, self.key)
