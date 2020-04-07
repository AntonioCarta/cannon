from __future__ import division
import os
import numpy as np
import fnmatch
import re

from cannon.rnn import LSTMLayer
from cannon.utils import cuda_move
import torch
from lxml import etree
from cannon.tasks import Dataset
import torch.nn.functional as F
from os import path
import Levenshtein
from tqdm import tqdm
from clockwork_lmn.container import ItemClassifier


def construct_ascii_path(ascii_path, f):
    primary_dir = f.split("-")[0]
    if f[-1].isalpha():
        sub_dir = f[:-1]
    else:
        sub_dir = f
    file_path = os.path.join(ascii_path, primary_dir, sub_dir, f + ".txt")
    return file_path


def construct_stroke_paths(strokes_path, f):
    primary_dir = f.split("-")[0]
    if f[-1].isalpha():
        sub_dir = f[:-1]
    else:
        sub_dir = f
    files_path = os.path.join(strokes_path, primary_dir, sub_dir)
    files = fnmatch.filter(os.listdir(files_path), f + "-*.xml")  # dash is crucial to obtain correct match!
    files = [os.path.join(files_path, fi) for fi in files]
    files = sorted(files, key=lambda x: int(x.split(os.sep)[-1].split("-")[-1][:-4]))
    return files


def parse_ascii_file(ascii_file):
    with open(ascii_file) as fp:
        lines = [t.strip() for t in fp.readlines() if t not in ['\r\n', '\n', ' \r']]

        # Find CSR lines
        idx_csr_start = [n for n, line in enumerate(lines) if line == "CSR:"][0]
        lines = lines[idx_csr_start + 1:]

        corrected_lines = []
        for line in lines:
            if "%" in line:  # Handle edge case with %%%%% meaning new line?
                line = re.sub('\%\%+', '%', line).split("%")
                line = [l.strip() for l in line]
                corrected_lines.extend(line)
            else:
                corrected_lines.append(line)
    return corrected_lines


def parse_strokes(strokes_files):
    x = []
    for stroke_file in strokes_files:
        with open(stroke_file) as fp:
            tree = etree.parse(fp)
            root = tree.getroot()
            # Get all the values from the XML
            # 0th index is stroke ID, will become up/down
            s = np.array([[i, float(Point.attrib['x']), float(Point.attrib['y']), float(Point.attrib['time'])]
                          for StrokeSet in root for i, Stroke in enumerate(StrokeSet) for Point in Stroke])

            def z_standardize(a):
                max, min = np.max(a), np.min(a)
                return (a - min) / (max - min)

            # normalization
            s[:, 1] = z_standardize(s[:, 1])   # x axis
            s[:, 2] = z_standardize(-s[:, 2])  # flip y axis
            s[:, 3] = z_standardize(s[:, 3])   # time

            # Get end of stroke points
            c = s[1:, 0] != s[:-1, 0]
            ci = np.where(c == True)[0]
            nci = np.where(c == False)[0]

            # set pen down
            s[0, 0] = 0
            s[nci, 0] = 0

            # set pen up
            s[ci, 0] = 1
            s[-1, 0] = 1
            x.append(s)
    return x


def fetch_iamondb_file(strokes_files, ascii_file):
    # A-Z, a-z, space, apostrophe, comma, period
    charset = list(range(65, 90 + 1)) + list(range(97, 122 + 1)) + [32, 39, 44, 46]
    tmap = {k: n + 1 for n, k in enumerate(charset)}
    tmap[0] = 0  # 0 for UNK/other

    def tokenize_ind(line):
        t = [ord(c) if ord(c) in charset else 0 for c in line]
        r = [tmap[i] + 1 for i in t]  # indices should start from 1 (0 is CTC blank)
        return r

    transcript_lines = parse_ascii_file(ascii_file)
    y = [np.zeros((len(li)), dtype='int16') for li in transcript_lines]
    for n, li in enumerate(transcript_lines):
        y[n][:] = tokenize_ind(li)

    x = parse_strokes(strokes_files)
    if len(x) != len(y):
        print("Dataset error: len(x) !+= len(y)!")
        raise ValueError()
    return x, y


def fetch_iamondb_from_list(data_dir, names):
    ascii_files = [construct_ascii_path(data_dir + "/ascii", f) for f in names]
    stroke_files = [construct_stroke_paths(data_dir + "/lineStrokes", f) for f in names]

    x_set = []
    y_set = []
    se = list(zip(stroke_files, ascii_files))
    for n, (strokes_n, ascii_n) in enumerate(se):
        if n % 100 == 0:
            print("Processing file %i of %i" % (n, len(stroke_files)))
        try:
            x, y = fetch_iamondb_file(strokes_n, ascii_n)
            y_set.extend(y)
            x_set.extend(x)
        except ValueError:
            print(f"Error on file {strokes_n, ascii_n}. Skipping it.")
    return x_set, y_set


def fetch_iamondb(data_dir):
    train_names = [f.strip() for f in open(data_dir + "/ids_train.txt", mode='r').readlines()]
    valid_names = [f.strip() for f in open(data_dir + "/ids_valid.txt", mode='r').readlines()]
    test_names = [f.strip() for f in open(data_dir + "/ids_test.txt", mode='r').readlines()]

    train_x, train_y = fetch_iamondb_from_list(data_dir, train_names)
    valid_x, valid_y = fetch_iamondb_from_list(data_dir, valid_names)
    test_x, test_y = fetch_iamondb_from_list(data_dir, test_names)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class IAMOnDB(Dataset):
    def __init__(self, data_dir, key, batch_size=32, debug=False) -> None:
        """
            IAMOnDB Handwritten character recognition.
            A many-to-one sequence problem, typically solved with RNN+CTC loss.
            refs: https://papers.nips.cc/paper/3213-unconstrained-on-line-handwriting-recognition-with-recurrent-neural-networks
        """
        super().__init__()
        self.key = key
        self.batch_size = batch_size

        if not path.exists(data_dir + f'/x_{key}.npy'):
            print("Loading from raw data.")
            f_ids = data_dir + f'/ids_{key}.txt'
            file_names = [f.strip() for f in open(f_ids, mode='r').readlines()]
            self.xs, self.ys = fetch_iamondb_from_list(data_dir, file_names)
            np.save(data_dir + f'/x_{key}.npy', self.xs, allow_pickle=True)
            np.save(data_dir + f'/y_{key}.npy', self.ys, allow_pickle=True)
        else:
            print("Loading numpy arrays.")
            self.xs = np.load(data_dir + f'/x_{key}.npy', allow_pickle=True)
            self.ys = np.load(data_dir + f'/y_{key}.npy', allow_pickle=True)

        self.xs = [cuda_move(torch.tensor(el)) for el in self.xs]
        self.ys = [cuda_move(torch.tensor(el)) for el in self.ys]

        if debug:
            self.xs = self.xs[:200]
            self.ys = self.ys[:200]

    def iter(self):
        n_samples = len(self.xs)
        if self.key == 'train':  # shuffle dataset
            idxs = torch.randperm(n_samples)
            self.xs = [self.xs[ii] for ii in idxs]
            self.ys = [self.ys[ii] for ii in idxs]
        for ii in range(0, len(self.xs), self.batch_size):
            x_batch = self.xs[ii:ii + self.batch_size]
            y_batch = self.ys[ii:ii + self.batch_size]
            bs = len(x_batch)
            x_t_max = max(el.shape[0] for el in x_batch)
            x = torch.zeros(x_t_max, bs, x_batch[0].shape[-1])
            y_t_max = max(el.shape[0] for el in y_batch)
            y = torch.zeros(bs, y_t_max)
            for bi, (x_el, y_el) in enumerate(zip(x_batch, y_batch)):
                x[:x_el.shape[0], bi] = x_el
                y[bi, :y_el.shape[0]] = y_el
            x = cuda_move(x)
            y = cuda_move(y).int()
            x_t_batch = cuda_move(torch.tensor([el.shape[0] for el in x_batch]).int())
            y_t_batch = cuda_move(torch.tensor([el.shape[0] for el in y_batch]).int())
            yield x, (y, x_t_batch, y_t_batch)

    def metric_score(self, batch, y_pred):
        """ Character Error Rate with best path decoding. """
        x, y_target = batch
        y_true, x_t_batch, y_t_batch = y_target

        e = 0
        for bi in range(x.shape[1]):
            yt_el = y_true[bi, :y_t_batch[bi]]
            yp_el = y_pred[:x_t_batch[bi], bi].argmax(dim=1)
            yp_el = yp_el[yp_el != 0]  # remove blanks

            st_el = ''.join([chr(el) for el in yt_el])
            sp_el = ''.join([chr(el) for el in yp_el])

            e += 1 - Levenshtein.distance(st_el, sp_el) / yt_el.shape[0]
        return e / x.shape[1]

    def loss_score(self, batch, y_pred):
        assert len(y_pred.shape) == 3
        x, y_target = batch
        y_true, x_t_batch, y_t_batch = y_target
        y_pred = F.log_softmax(y_pred, dim=2)
        err = F.ctc_loss(y_pred, y_true, x_t_batch, y_t_batch)
        return err

    def __str__(self):
        return f"IAMOnDB Dataset (key={self.key}, bs={self.batch_size})"


if __name__ == '__main__':
    data_dir = '/home/carta/data/iamondb'

    # # Load model and compute WER
    # valid_data = IAMOnDB(data_dir, key='valid')
    # model = ItemClassifier(
    #     rnn=LSTMLayer(4, 48),
    #     hidden_size=48 * 9,
    #     output_size=58
    # )
    # model.load_state_dict(torch.load('./logs/iamondb/lstm_gs0/k_0/best_model.pt'))
    # model = cuda_move(model)
    # acc = 0
    # bi = 0
    # for batch in tqdm(valid_data.iter()):
    #     y_pred = model(batch[0])
    #     acc += valid_data.metric_score(batch, y_pred)
    #     bi += 1
    # acc = acc / bi
    # print(f"ACC: {acc}")
    # exit()

    # Dataset statistics
    data = fetch_iamondb(data_dir)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data
    train_x = np.vstack(train_x)

    print(f"Time dim (avg, std): {np.mean(train_x[:, 1]), np.std(train_x[:, 1])}")
    print(f"x dim (avg, std): {np.mean(train_x[:, 2]), np.std(train_x[:, 2])}")
    print(f"y dim (avg, std): {np.mean(train_x[:, 3]), np.std(train_x[:, 3])}")

    print(f"Time dim (min, max): {np.min(train_x[:, 1]), np.max(train_x[:, 1])}")
    print(f"x dim (min, max): {np.min(train_x[:, 2]), np.max(train_x[:, 2])}")
    print(f"y dim (min, max): {np.min(train_x[:, 3]), np.max(train_x[:, 3])}")

    # Class Loader and loss computation
    valid_data = IAMOnDB(data_dir, key='valid')
    e = 0
    for batch in valid_data.iter():
        x, (y, x_t_batch, y_t_batch) = batch
        rand_pred = cuda_move(torch.randn(x.shape[0], x.shape[1], 58))
        e += valid_data.loss_score(batch, rand_pred)
    print(f"err: {e}")

    # Data Loader
    data = fetch_iamondb(data_dir)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data

    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    assert len(test_x) == len(test_y)
    # 5364,  1,438,  1,518  and  3,859  written  lines  taken  from 775,  192,  216 and 544 forms
    print(f"# of features: {train_x[0].shape[1]}")
    print(f"classes: from {np.min([np.min(el) for el in train_y])} to {np.max([np.max(el) for el in train_y])}")

    print(f"train: {len(train_x)}")
    print(f"valid: {len(valid_x)}")
    print(f"test: {len(test_x)}")

    len_seqs = [x.shape[0] for x in train_x]
    print(f"input len avg: {np.mean(len_seqs):.2f}, min: {np.min(len_seqs)}, max: {np.max(len_seqs)}")

    len_seqs = [y.shape[0] for y in train_y]
    print(f"output len avg: {np.mean(len_seqs):.2f}, min: {np.min(len_seqs)}, max: {np.max(len_seqs)}")

    print("Done.")
