"""
    Preprocess TIMIT into a list of segmented phonemes.
"""
import soundfile as sf
import numpy as np
import python_speech_features as psf
import pickle
import os
import torch
from pandas import read_csv
from cannon.tasks import Dataset
import random
from cannon.utils import cuda_move
import torch.nn.functional as F


class TIMITFramewisePhonemeClassification(Dataset):
    def __init__(self, base_timit, mode, batch_size, debug=False):
        """
        Phoneme classification with segmented samples extracted from TIMIT.

        Args:
            mode:
        """
        assert mode in {'train', 'valid', 'test'}
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        lazy_preprocess(base_timit)
        if debug:
            print("Subsampling data for debugging.")
            with open(os.path.join(base_timit, f'framewise_phoneme/debug_split.pkl'), 'rb') as f:
                self.fns, self.xs, self.ys = pickle.load(f)
        else:
            with open(os.path.join(base_timit, f'framewise_phoneme/{mode}_split.pkl'), 'rb') as f:
                self.fns, self.xs, self.ys = pickle.load(f)

    def iter(self):
        n_samples = len(self.xs)
        if self.mode == 'train':
            # shuffle dataset
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
            y = torch.zeros(t_max, bs, dtype=torch.long)
            for bi, (x_el, y_el) in enumerate(zip(x_batch, y_batch)):
                x[:x_el.shape[0], bi] = x_el
                y[:x_el.shape[0], bi] = y_el
            x = cuda_move(x)
            y = cuda_move(y)
            t_batch = cuda_move(t_batch)
            yield x, (y, t_batch)

    def metric_score(self, y_pred, y_target):
        assert len(y_pred.shape) == 3
        y_true, t_batch = y_target
        yp = []
        yt = []
        for bi in range(y_pred.shape[1]):
            yp.append(y_pred[:t_batch[bi], bi])
            yt.append(y_true[:t_batch[bi], bi])
        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)
        acc = (yp.argmax(dim=1) == yt).float().mean()
        return acc

    def loss_score(self, y_pred, y_target):
        assert len(y_pred.shape) == 3
        y_true, t_batch = y_target
        yp = []
        yt = []
        for bi in range(y_pred.shape[1]):
            yp.append(y_pred[:t_batch[bi], bi])
            yt.append(y_true[:t_batch[bi], bi])
        yp = torch.cat(yp, dim=0)
        yt = torch.cat(yt, dim=0)
        err = F.cross_entropy(yp, yt)
        return err


all_phonemes = [
    "b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx",
    "q", "jh", "ch", "s", "sh", "z", "zh", "f", "th", "v", "dh", "m", "n", "ng",
    "em", "en", "eng", "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih",
    "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux",
    "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"
]  # 61 phonemes


def lazy_preprocess(base_timit):
    if not os.path.exists(os.path.join(base_timit, 'framewise_phoneme', 'train_segm.pkl')):
        print("preprocessing training set.")
        data = preprocess_timit(base_timit, 'train')
        with open(os.path.join(base_timit, 'framewise_phoneme', 'train_segm.pkl'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(os.path.join(base_timit, 'framewise_phoneme', 'test_segm.pkl')):
        print("preprocessing validation set.")
        data = preprocess_timit(base_timit, 'test')
        with open(os.path.join(base_timit, r'framewise_phoneme/test_segm.pkl'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(os.path.join(base_timit, 'framewise_phoneme', 'train_split.pkl')):
        print("splitting training/validation set.")
        save_normalized_splits(base_timit, 0.2)


def process_wav(path_wav, path_phoneme):
    """
    Process each wav dividing it into frames and computing MFCC and the corresponding phoneme for each frame.

    Args:
        path_wav (str): wav file with speech data
        path_phoneme (str): file with segmented phonemes

    Returns:
        (features, phonemes): tuple containing the sequence of frames and the sequence of corresponding phonemes.
    """
    data, sample_rate = sf.read(path_wav)  # np.array of shape (len,), int (16000)

    file_content = read_csv(path_phoneme, sep=' ', header=None, names=['start', 'end', 'phn'])
    phonemes = list(map(lambda ph: all_phonemes.index(ph), file_content['phn'].values.tolist()))
    times = file_content[['start', 'end']].values.tolist()

    # the first feature is the log energy
    features = psf.mfcc(data, sample_rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97,
                        appendEnergy=True, winfunc=np.hamming)

    # the first delta is related to the log energy
    # the second parameter is #frames before and after current to use in the computation
    deltas = psf.delta(features, 2)

    features = torch.from_numpy(np.concatenate((features, deltas), axis=1)).float()

    T_frames = features.shape[0]
    T_samples = data.shape[0]
    labels = torch.zeros(features.shape[0], dtype=torch.long) - 1
    si = 0
    for ti, pi in zip(times, phonemes):
        _, ei = ti
        ei = round(ei / T_samples * T_frames)
        labels[si:ei] = pi
        si = ei

    assert torch.sum(labels[:ei] == -1) == 0
    return features[:ei], labels[:ei]


def preprocess_timit(base_timit, mode):
    """
    Store timit features and labels in a list on disk

    The list has the following structure:
    [
        LIST OF FILENAMES, LIST OF FEATURES, LIST OF LABELS
    ]

    FEATURES is a list of feature tensor
    LABELS is a tensor of phonemes indexes
    """
    assert mode in {'train', 'test'}
    dump_filename = f'framewise_phoneme/{mode}_frames.pkl'
    folder = mode.upper()

    data_path = os.path.join(base_timit, folder)
    file_names, frame_list, phoneme_list = [], [], []
    for dir, subdirs, files in os.walk(data_path):
        print('Entering directory ', dir)
        for f in files:
            if f.endswith('.WAV') and not f.startswith('SA'):
                file_name = f.split('.')[0]
                features, phonemes = process_wav(os.path.join(dir, f), os.path.join(dir, file_name + '.PHN'))

                file_names.append(os.path.join(dir, file_name))
                frame_list.append(features)
                phoneme_list.append(phonemes)
    return file_names, frame_list, phoneme_list


def save_normalized_splits(base_timit, p):
    print(f"Splitting TRAIN and VALID with p={p}")
    fname = os.path.join(base_timit, 'framewise_phoneme/train_segm.pkl')
    with open(fname, 'rb') as f:
        fns, xs, ys = pickle.load(f)

    means = torch.cat(xs, dim=0).mean(dim=0)
    stds = torch.cat(xs, dim=0).std(dim=0)

    def normalize_frame_sequences(l):
        new_l = []
        for seq in l:
            new_seq = (seq - means) / stds
            new_l.append(new_seq)
        return new_l

    xs = normalize_frame_sequences(xs)
    f_tr, x_tr, y_tr = [], [], []
    f_val, x_val, y_val = [], [], []
    for ii in range(len(xs)):
        if random.random() < p:
            f_val.append(fns[ii])
            x_val.append(xs[ii])
            y_val.append(ys[ii])
        else:
            f_tr.append(fns[ii])
            x_tr.append(xs[ii])
            y_tr.append(ys[ii])

    print(f"{len(x_val)}/{len(x_val) + len(x_tr)} samples in validation.")
    with open(os.path.join(base_timit, 'framewise_phoneme/train_split.pkl'), 'wb') as f:
        pickle.dump([f_tr, x_tr, y_tr], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base_timit, 'framewise_phoneme/valid_split.pkl'), 'wb') as f:
        pickle.dump([f_val, x_val, y_val], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base_timit, 'framewise_phoneme/debug_split.pkl'), 'wb') as f:
        pickle.dump([f_val[:100], x_val[:100], y_val[:100]], f, protocol=pickle.HIGHEST_PROTOCOL)

    fname = os.path.join(base_timit, 'framewise_phoneme/test_segm.pkl')
    with open(fname, 'rb') as f:
        fns, xs, ys = pickle.load(f)
    xs = normalize_frame_sequences(xs)
    print(f"{len(xs)} samples in test.")
    with open(os.path.join(base_timit, 'framewise_phoneme/test_split.pkl'), 'wb') as f:
        pickle.dump([fns, xs, ys], f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    base_dir = '/home/carta/data/timit/TIMIT'

    lazy_preprocess(base_dir)


    # split_train_val_test(base_dir, 0.2)
    # exit(0)
    #
    # # normalizations = get_mean_std(base_dir)
    # normalizations = (-5.062385082244873, 13.341102600097656, -4.6380391120910645, 12.983874320983887)
    # print(normalizations)

    # preprocess_timit(base_dir, 'train', normalizations)
    # preprocess_timit(base_dir, 'test', normalizations)
    # td_train = load_timit(base_dir, 'train')
    # td_test = load_timit(base_dir, 'test')
    #
    # x, y = td_train[1], td_train[2]
    # for i in range(len(x)):
    #     f, l = x[i], y[i]
    #     print(len(f), len(l))
    #     try:
    #         assert(len(f) == l.size(0))
    #     except:
    #         print(td_train[0][i])
    #
    # print(len(td_train))
    # count_phonemes(base_dir, 'train')
    # {'DR1': 11635, 'DR2': 23543, 'DR3': 23259, 'DR4': 21036, 'DR5': 21883, 'DR6': 10824, 'DR7': 24007, 'DR8': 6723}
