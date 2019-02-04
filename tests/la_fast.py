"""
    Linear Autoencoder for Sequences
    Reduced better memory footprint
    - block Xi
    - block R
"""
from tasks import PianoRollData
import numpy as np
from cannon.la.svd_la import LinearAutoencoder
import numpy.linalg as la
import scipy.sparse as sparse
import time


def get_Xi_block(data, n_block):
    len_samples = [len(el) for el in data]
    tot_len = sum(len_samples)
    Xhii = np.zeros((tot_len, data[0][0].shape[0]))

    sum_prev_samples = 0
    for sample_i, sample in enumerate(data):
        for t_step in range(len(sample) - n_block):
            Xhii[sum_prev_samples + t_step + n_block, :] = sample[t_step]
        sum_prev_samples += len_samples[sample_i]
    return Xhii


def Svd_single_column(data, n_components=10, verbose=False):
    """ Compute the SVD for big matrices, approximating the result.

    Args:
        data: input matrix.
        sample_dim: dimension of a single slice.
        n_components: number of principal components to return as output.
    """
    from cannon.la.big_svd import KeCSVD
    num_slices = max([len(el) for el in data])
    last_slice = get_Xi_block(data, num_slices)

    # compute svd for the last slice, and then repeat this process for each slice concatenated with the previous result.
    v, s, u_t = KeCSVD(last_slice, n_components)
    for i in reversed(range(num_slices)):
        if verbose:
            print("slice", i, " of ", num_slices)
        curr_vs = v @ s
        curr_slice = get_Xi_block(data, i)
        curr_vs = np.hstack((curr_slice, curr_vs))
        v, s, u_t = KeCSVD(curr_vs, n_components)
    return v, s, u_t


def build_R_block(seq_len):
    """ Create the R matrix as a sparse matrix.
    Args:
        len_samples: length of sequences contained in Xhi
    """
    Rrow = []
    Rcol = []
    Rdata = []
    for i in range(1, seq_len):
        Rrow.append(i)
        Rcol.append(i - 1)
        Rdata.append(1)
    return sparse.coo_matrix((Rdata, (Rrow, Rcol)), shape=(seq_len, seq_len))


def vt_R_v_block_multiplication(V, len_samples):
    res = np.zeros_like(V.T)

    len_sum = 0
    for seq_len in len_samples:
        R = build_R_block(seq_len)
        v_block = V[len_sum: len_sum + seq_len]
        vtr_block = v_block.T @ R
        res[:, len_sum: len_sum + seq_len] = vtr_block
        len_sum += seq_len

    res = res @ V
    return res


class FastLA(LinearAutoencoder):
    def fit(self, data, approximate, verbose=False):
        if self.A is not None:
            print("linear autoencoder has been already trained.")

        len_samples = [len(el) for el in data]
        batch_size = len(len_samples)
        n = data[0].shape[0]
        k = data[0].shape[1]
        self.p = min(self.p, k*n, batch_size*n)
        p = self.p

        # compute mean
        curr_mean = 0
        for el in data:
            curr_mean += el.shape[0] * np.mean(el, axis=0)
        self.mean = curr_mean / sum(len_samples)

        # mean shift
        for i in range(len(data)):
            data[i] = data[i] - self.mean

        if verbose:
            print("computing SVD decomposition.")
        if approximate:
            V, s, Uh = Svd_single_column(data, self.p, verbose=verbose)
            s = np.diag(s).copy()
        else:
            V, s, Uh = la.svd(data, full_matrices=False)

        V = V[:, :p]
        s = s[:p]
        Uh = Uh[:p, :]

        s[s < self.epsilon] = 0
        inv_s = s.copy()
        inv_s[s > self.epsilon] = 1 / s[s > self.epsilon]
        vtrv = vt_R_v_block_multiplication(V, len_samples)
        Q = np.diag(inv_s) @ vtrv @ np.diag(s)

        self.A = Uh.T[:k, :].T
        self.B = Q
        self.sigma = s


if __name__ == '__main__':
    f_data = '../linear-memory/data/MIDI/piano-roll/JSB Chorales.pickle'
    data = PianoRollData(f_data, key='train')
    t_max = 200
    n = 800

    X = []
    for i in range(len(data)):
        el = data[i]
        X.append(el[0][:t_max].numpy())

    lin_fast = FastLA(p=50)
    lin_old = LinearAutoencoder(p=50)

    t_start = time.time()
    lin_fast.fit(X, approximate=True, verbose=True)
    t_end_fast = time.time() - t_start

    t_start = time.time()
    lin_old.fit(X, approximate=True, verbose=True)
    t_end_old = time.time() - t_start

    error_norm_A = la.norm(lin_fast.A - lin_old.A) / la.norm(lin_old.A)
    error_norm_B = la.norm(lin_fast.B - lin_old.B) / la.norm(lin_old.B)
    print("error A: {}, error B: {}".format(error_norm_A, error_norm_B))
    print("total time t_fast: {}, t_old: {}".format(t_end_fast, t_end_old))
    assert error_norm_A < 1.e-5 and error_norm_B < 1.e-5

