import torch
import random
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import os
import soundfile as sf
from python_speech_features import mfcc
from cannon.tasks import Dataset
import torch.nn.functional as F
from cannon.utils import cuda_move


class TimitCommonSuffix(Dataset):
    def __init__(self, data_folder, key, batch_size=25):
        """
        Timit Word task as defined in the Clockwork RNN paper.

        Args:
            data_folder: root directory for TIMIT data
            key: one of 'train', 'valid', 'test'
            batch_size: for class balancing reasons it can be either 1 or 25
        """
        super().__init__()
        assert batch_size in {1, 25}
        assert key in {'train', 'test', 'valid', 'noisy_valid'}
        self.batch_size = batch_size
        if key == 'test':
            data_file = data_folder + f'preprocessed/{key}.pt'
        else:
            data_file = data_folder + 'preprocessed/train.pt'
        if not os.path.exists(data_file):
            save_timit_data(data_folder, self.batch_size, numcep=13)
        self.X, self.y = torch.load(data_file)
        self.n_classes = len(self.y)
        self.key = key
        self.noise_std = 0.0
        if self.key == 'train' or self.key == 'noisy_valid':
            self.noise_std = 0.6

    def iter(self):
        it = timit_generator(self.X, self.y, self.n_classes, self.batch_size, self.noise_std)
        for el in it:
            yield [cuda_move(x) for x in el]

        if self.key == 'noisy_valid':
            for _ in range(4):
                it = timit_generator(self.X, self.y, self.n_classes, self.batch_size, self.noise_std)
                for el in it:
                    yield [cuda_move(x) for x in el]

    def loss_score(self, batch, y_pred):
        _, y_target, t_mask = batch
        yp = []
        for bi in range(y_target.shape[0]):
            yp.append(y_pred[t_mask[bi] - 1, bi])
        y_pred = torch.stack(yp, dim=0)
        return F.cross_entropy(y_pred, y_target)

    def metric_score(self, batch, y_pred):
        _, y_target, t_mask = batch
        yp = []
        for bi in range(y_target.shape[0]):
            yp.append(y_pred[t_mask[bi] - 1, bi])
        y_pred = torch.stack(yp, dim=0)
        return (y_pred.argmax(dim=1) == y_target).float().mean()


def save_timit_data(data_folder, batch_size=25, numcep=13):
    vs_ratio = .285714286
    preproc_folder = data_folder + 'preprocessed/'
    tr_x, tr_y, vs_x, vs_y, X, C = build_dataset_timit(data_folder, vs_ratio, batch_size, numcep)
    os.makedirs(preproc_folder, exist_ok=True)
    with open(preproc_folder + 'train.pt', 'wb') as f:
        torch.save([tr_x, tr_y], f)
    with open(preproc_folder + 'test.pt', 'wb') as f:
        torch.save([vs_x, vs_y], f)
    with open(preproc_folder + 'all.pt', 'wb') as f:
        torch.save([X, C], f)


def shuffle(X, Y):
    index_shuf = list(range(len(X)))
    random.shuffle(index_shuf)
    Xs = [X[j] for j in index_shuf]
    if isinstance(X, type(torch.tensor(0))):
        Xs = torch.stack(Xs)

    if Y is None:
        return Xs
    else:
        Ys = [Y[j] for j in index_shuf]
        if isinstance(Y, type(torch.tensor(0))):
            Ys = torch.stack(Ys)
        return Xs, Ys


def build_dataset_timit(main_folder, split_ratio_vs, batch_size, numcep=13):
    train_folder = main_folder + 'TIMIT/TRAIN/'
    test_folder = main_folder + 'TIMIT/TEST/'

    data = pd.read_csv(main_folder + "clockwork-task-words.csv", header=0, delimiter=",")
    codes = data.Code.values.copy()
    words = data.Word.values.copy()
    classes = data.Class.values.copy()
    print(f"FOUND WORDS: {words}")
    print(f"FOUND WORDS: {classes}")
    wavs = [[] for _ in range(25)]
    for folder in [train_folder, test_folder]:
        for dir_x in os.listdir(folder):
            if os.path.isdir(folder + dir_x):
                for personcode in os.listdir(folder + dir_x):
                    person_folder = folder + dir_x + '/' + personcode + '/'
                    if os.path.isdir(person_folder):
                        for code in os.listdir(person_folder):
                            idxs = np.where(code[0:-4] == codes)[0] if code[-4:] == '.WAV' else np.array([])
                            for i in idxs:
                                # assert code[0:-4] == codes[i], "The code found is not the current one"
                                found_code = codes[i]
                                found_word = words[i]
                                found_class = classes[i]
                                wrdfile = pd.read_csv(person_folder + found_code + ".WRD", names=['start', 'end', 'word'], delimiter=" ")
                                sample_start = -1
                                sample_end = -1
                                for start, end, word in zip(wrdfile.start, wrdfile.end, wrdfile.word):
                                    if found_word == word.lower():
                                        sample_start = start
                                        sample_end = end
                                        break
                                # assert sample_start != -1, f"Word {word} not found in corresponding {person_folder+found_code}.WRD file"
                                signal, samplerate = sf.read(person_folder + found_code + '.WAV')
                                signal = signal[sample_start:sample_end]
                                wavs[found_class - 1].append((signal, samplerate))

    Y = []
    for c, sublist in enumerate(wavs):
        for _ in sublist:
            Y.append(c)
    X = [item for sublist in wavs for item in sublist]
    X, longest_sequence = _to_mfcc(X, numcep)
    tr_x, tr_y, vs_x, vs_y, max_c = _split_class_buckets(X, Y, split_ratio_vs, batch_size)
    return tr_x, tr_y, vs_x, vs_y, X, Y


def build_generators_timit(tr_x, tr_y, vs_x, vs_y, batch_size, noise_std, n_classes):
    return lambda: timit_generator(tr_x, tr_y, n_classes, batch_size, noise_std=noise_std), \
        lambda: timit_generator(vs_x, vs_y, n_classes, batch_size, noise_std=0.0)


def _to_mfcc(X, numcep):
    X = [mfcc(Xi[0], Xi[1], winlen=0.025, winstep=0.01, preemph=0.97, numcep=numcep, appendEnergy=True) for Xi in X if len(Xi[0]) > 0]
    longest_sequence = -1
    for Xi in X:
        longest_sequence = Xi.shape[0] if Xi.shape[0] > longest_sequence else longest_sequence
    split_sizes = [len(t) for t in X]
    X = np.concatenate(X, axis=0)
    X = pre.scale(X, axis=0)
    X = list(torch.split(torch.tensor(X, dtype=torch.float), split_sizes, dim=0))
    return X, longest_sequence


def _split_class_buckets(X, C, split_ratio_vs, batch_size):
    c_size = round(len(X) / (max(C)+1))
    c_tr_size = round(c_size * (1 - split_ratio_vs))
    c_vs_size = round(c_size * split_ratio_vs)
    tr_x, tr_y, vs_x, vs_y = [], [], [], []
    if batch_size == 1:
        for c in range(max(C)+1):
            x = X[c*c_size:c*c_size+c_tr_size+c_vs_size]
            y = C[c*c_size:c*c_size+c_tr_size+c_vs_size]
            x, y = shuffle(x, y)
            tr_x = tr_x + x[:c_tr_size]
            tr_y = tr_y + y[:c_tr_size]
            vs_x = vs_x + x[c_tr_size:c_tr_size+c_vs_size]
            vs_y = vs_y + y[c_tr_size:c_tr_size+c_vs_size]
    else:
        for c in range(max(C)+1):
            x = X[c*c_size:c*c_size+c_tr_size+c_vs_size]
            y = C[c*c_size:c*c_size+c_tr_size+c_vs_size]
            x, y = shuffle(x, y)
            tr_x.append(x[:c_tr_size])
            tr_y.append(y[:c_tr_size])
            vs_x.append(x[c_tr_size:c_tr_size+c_vs_size])
            vs_y.append(y[c_tr_size:c_tr_size+c_vs_size])
    return tr_x, tr_y, vs_x, vs_y, max(C)+1


def shuffle_pair(x, y):
    idx = torch.randperm(x.shape[0])
    return x[idx], y[idx]


def timit_generator(X, y, n_classes, batch_size, noise_std=0.6):
    assert batch_size > 0 and n_classes > 0 and (batch_size == 1 or batch_size % n_classes == 0), \
        f"batch size must be a multiple of the number of classes, batch_size {batch_size}, max_c {n_classes}"
    if batch_size == 1:
        X_new, y_new = [], []
        for cx, cy, in zip(X, y):
            X_new.extend(cx)
            y_new.extend(cy)
        idx = list(range(len(X_new)))
        random.shuffle(idx)
        X = [X_new[i] for i in idx]
        y = [y_new[i] for i in idx]
    else:
        for i, Xc in enumerate(X):
            index_shuf = list(range(len(Xc)))
            random.shuffle(index_shuf)
            X[i] = [Xc[j] for j in index_shuf]
            y[i] = [y[i][j] for j in index_shuf]

    start = 0
    remaining = len(X) if batch_size == 1 else len(X[0])
    while remaining > 0:
        if batch_size == 1:
            step = 1
            Xb = X[start:start+step]
            yb = torch.tensor(y[start:start+step])
            tb = torch.tensor([el.shape[0] for el in Xb])
        else:
            step = min(int(batch_size / n_classes), remaining)
            Xb = []
            yb = []
            for c, Xc in enumerate(X):
                Xb.extend(Xc[start:start+step])
                for j in range(step):
                    yb.append(c)
            tb = torch.tensor([el.shape[0] for el in Xb])
            yb = torch.tensor(yb)
            Xb, yb = shuffle(Xb, yb)

        if noise_std > 0:
            Xb = _add_noise(Xb, std=noise_std)
        Xb = _add_padding_timit(Xb)
        Xb = Xb.permute(1, 0, 2)  # We want: sequence x batch x features
        start += step
        remaining -= step
        yield Xb, yb, tb


def _add_noise(X, std):
    X = [Xi + torch.empty_like(Xi).normal_(mean=0.0, std=std) for Xi in X]
    return X


def _add_padding_timit(X):
    max_len = max(el.shape[0] for el in X)
    Xpad = torch.zeros(len(X), max_len, len(X[0][0]))
    for i, Xi in enumerate(X):
        Xpad[i][0:len(Xi)] = Xi
    return Xpad


if __name__ == '__main__':
    train = TimitCommonSuffix('data/timit/', 'train')
    valid = TimitCommonSuffix('data/timit/', 'train')
    test = TimitCommonSuffix('data/timit/', 'train')

    print('')
    for i, b in enumerate(train.iter()):
        x, y, t = b
        yp = torch.zeros(x.shape[0], x.shape[1], train.n_classes)
        for bi in range(y.shape[0]):
            yp[t[bi] - 1, bi, y[bi]] = 1 * i

        a = train.metric_score(b, yp)
        e = train.loss_score(b, yp)
        print(f"{a}, {e}")
    print(f"{i + 1} batches")
