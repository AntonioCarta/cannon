import numpy as np
import scipy.linalg as la
from scipy import sparse
from sklearn.exceptions import NotFittedError
import pickle
import math
import fbpca
try:
    from numba import njit, prange
    is_numba_available = True
except ImportError:
    print("Numba not installed.")
    is_numba_available = False
from scipy.sparse import linalg as splinalg


if is_numba_available:
    @njit(fastmath=True)
    def xi_seq_rmatvec(seq, v, res=None):
        """ Implicit matrix multiplication Xi_seq.T @ v
        Args
            seq: single sequence [time x features]
            v: vector [features,]
            res: resulting vector
        Returns:
            Xi_seq @ v
        """
        t, f = seq.shape
        assert v.shape[0] == t
        if res is None:
            res = np.zeros(t*f)
        for iit in prange(t):
            tmp = (seq[:t-iit].T @ v[iit:]).reshape(-1)
            res[f*iit: f*iit+f] += tmp
        return res


    @njit(fastmath=True, parallel=True)
    def xi_data_rmatvec(data, v):
        """ Implicit matrix multiplication Xi_data.T @ v
        Args
            data: list of sequences [time x features]
            v: vector [features,]
        Returns:
            Xi_data @ v
        """
        len_samples = np.asarray([data[i].shape[0] for i in range(len(data))])
        len_max = max(len_samples)
        res = np.zeros(len_max*data[0].shape[1])
        curr_idx = 0
        for i in range(len(data)):
            seq = data[i]
            t_seq, f = seq.shape
            xi_seq_rmatvec(seq, v[curr_idx: curr_idx+t_seq], res)
            curr_idx += t_seq
        assert curr_idx == np.sum(len_samples)
        return res


    @njit(fastmath=True)
    def xi_seq_matvec(seq, v1, res=None):
        """ Implicit matrix multiplication Xi_seq @ v
        Args
            seq: single sequence [time x features]
            v: vector [features,]
        Returns:
            Xi_seq @ v
        """
        t, f = seq.shape
        assert v1.shape[0] == t*f
        v = v1.reshape(t, f)
        if res is None:
            res = np.zeros(t)
        for iiv in range(v.shape[0]):
            res[iiv] = 0
            for k in range(iiv + 1):
                res[iiv] += seq[iiv - k].T @ v[k]
        return res


    @njit(fastmath=True, parallel=False)
    def xi_data_matvec(data, v):
        """ Implicit matrix multiplication Xi_data @ v
        Args
            data: list of sequences [time x features]
            v: vector [features,]
        Returns:
            Xi_data @ v
        """
        len_samples = np.asarray([data[i].shape[0] for i in range(len(data))])
        sum_len = np.sum(len_samples)
        res = np.zeros(sum_len)

        curr_idx = 0
        idxs = []
        for i in range(len(data)):
            idxs.append(curr_idx)
            t_seq, _ = data[i].shape
            curr_idx += t_seq

        for i in prange(len(data)):
            curr_idx = idxs[i]
            seq = data[i]
            t_seq, f = seq.shape
            xi_seq_matvec(seq, v[:t_seq*f], res[curr_idx: curr_idx+t_seq])
        return res


def get_Xi_block(data, n_block):
    len_samples = [len(el) for el in data]
    tot_len = sum(len_samples)
    Xhii = np.zeros((tot_len, data[0][0].shape[0]), dtype=np.float32)

    sum_prev_samples = 0
    for sample_i, sample in enumerate(data):
        if len(sample) - n_block > 0:
            Xhii[sum_prev_samples + n_block: sum_prev_samples + sample.shape[0]] = sample[:len(sample) - n_block]
        sum_prev_samples += len_samples[sample_i]
    return Xhii


def get_Xi_range(data, idx_start, idx_end):
    Xi_blocks = []
    for ii in range(idx_start, idx_end + 1):
        Xi_blocks.append(get_Xi_block(data, ii))
    return np.hstack(Xi_blocks)


def Svd_single_column(data, n_components=10, k=1, verbose=False):
    """ Compute the SVD for big matrices, approximating the result.

    Args:
        data: input matrix.
        sample_dim: dimension of a single slice.
        n_components: number of principal components to return as output.
    """
    num_slices = max([len(el) for el in data])

    if k >= num_slices:
        if verbose:
            print("k > num_slices. computing the SVD in a single step.")
        curr_slice = []
        curr_slice = get_Xi_range(data, 0, num_slices - 1)
        V, s, Uh = np.linalg.svd(curr_slice)
        return V[:, :n_components], np.diag(s[:n_components]), Uh[:n_components, :]

    remaining = num_slices - math.floor(num_slices / k) * k
    if remaining == 0:
        remaining = k

    last_slice = get_Xi_range(data, num_slices - remaining, num_slices - 1)
    if verbose:
        print("slice: ", num_slices - remaining, " to ", num_slices - 1)
    # compute svd for the last slice, and then repeat this process for each slice concatenated with the previous result.
    v, s, u_t = KeCoSVD(last_slice, n_components)
    del last_slice
    for i in reversed(range(0, num_slices - remaining, k)):
        curr_vs = v @ s
        del v, s, u_t

        curr_slice = get_Xi_range(data, i, i + k - 1)
        if verbose:
            print("slice: ", i, " to ", i + k - 1)

        curr_vs = np.hstack((curr_slice, curr_vs))
        v, s, u_t = KeCoSVD(curr_vs, n_components)
        del curr_vs
    return v, s, u_t


def indirectSVD(M, p):
    """
    SVD decomposition.

    Args:
        M: input matrix
        p: number of principal components.
    """
    Q, R = np.linalg.qr(M)
    v_r, s, u_t = np.linalg.svd(R)
    v = Q.dot(v_r)

    s = s[0:p]
    v = v[:, 0:p]
    u_t = u_t[0:p, :]
    return v, np.diag(s), u_t


def KeCoSVD(M, p=10, min_sigma=0.000001):
    c = 0
    if M.shape[0] <= M.shape[1]:
        # svd on kernel matrix
        Kernel = M @ M.transpose()
        v, Ssqr, _ = indirectSVD(Kernel, p)
        s = np.sqrt(Ssqr)
        u_t = np.linalg.pinv(s) @ v.transpose() @ M
    else:
        # svd on covariance matrix
        Cov = M.transpose() @ M
        _, Ssqr, u_t = indirectSVD(Cov, p)
        s = np.sqrt(Ssqr)
        v = M @ u_t.transpose() @ np.linalg.pinv(s)
    C = s.shape[0]

    for c in reversed(range(C)):
        if s[c, c] > min_sigma:
            break
    # TODO: this can change the dimension.
    # either fill with zeros or output some warning.
    v[:, c:] = 0
    s[c:, c:] = 0
    u_t[c:, :] = 0
    return v, s, u_t


def build_xhi_matrix(data, verbose=True):
    """ Create the Xhi matrix (as a dense matrix).
    Args:
        data: list of sequences (even with different lengths)
    Returns:
        data matrix with reversed subsequences as rows.
    """
    n_features = data[0].shape[-1]
    len_samples = [len(el) for el in data]
    tot_len = sum(len_samples)
    max_len = max(len_samples)
    Xhi = np.zeros((tot_len, max_len * n_features))

    sum_prev_samples = 0
    for sample_i, sample in enumerate(data):
        if sample_i % 100 == 0 and verbose:
            print(f"sample {sample_i}")
        for t_step in range(len(sample)):
            it = min(len(sample) - t_step, max_len)
            for offset in range(it):
                row_idx = sum_prev_samples + t_step + offset
                col_idx = offset * n_features
                Xhi[row_idx, col_idx:col_idx + n_features] = sample[t_step]
        sum_prev_samples += len_samples[sample_i]
    return Xhi


def build_R(len_samples):
    """ Create the R matrix as a sparse matrix.
    Args:
        len_samples: length of sequences contained in Xhi
    """
    samplePos = 1
    Rrow = []
    Rcol = []
    Rdata = []
    for len_i in len_samples:
        for i in range(samplePos, samplePos + (len_i - 1)):
            Rrow.append(i)
            Rcol.append(i - 1)
            Rdata.append(1)
        samplePos = samplePos + len_i
    return sparse.coo_matrix((Rdata, (Rrow, Rcol)), shape=(sum(len_samples), sum(len_samples)))


def get_la_weights(vl, sl, u_l, len_samples, n_features, epsilon=1.e-6):
    """ Given the svd decomposition of the Xhi matrix, this function compute the linear autoencoder matrices A and B.
    The number of hidden units is determined by the number of SVD principal components.
    """
    inv_s = sl.copy()
    inv_s[sl > epsilon] = 1 / sl[sl > epsilon]
    vtrv = vt_R_v_block_multiplication(vl, len_samples)
    Q = np.diag(inv_s) @ vtrv @ np.diag(sl)

    A = u_l.T[:n_features, :].T
    B = Q
    return np.asarray(A), np.asarray(B), sl.shape[0]


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


class LinearAutoencoder:
    def __init__(self, p, epsilon=1e-7, whiten=True):
        self.epsilon = epsilon
        self.whiten = whiten
        self.p = p
        # parameters
        self.A = None
        self.B = None
        self.mean = 0.0
        self.sigma = None

    def fit(self, data, svd_algo='fb_pca', approx_k=1, t_max=None, verbose=False):
        """
        Fit the Linear Autoencoder using SVD-based training.

        Args:
            data: list of sequences (possibly with different lengths)
        """
        if verbose:
            print(f"svd_algo = {svd_algo}")
        if self.A is not None:
            print("linear autoencoder has been already trained.")

        if t_max:
            new_data = []
            for el in data:
                new_data.append(el[:t_max])
            data = new_data
        len_samples = [len(el) for el in data]
        batch_size = len(len_samples)
        n = data[0].shape[0]
        k = data[0].shape[1]
        # self.p = min(self.p, k*n, batch_size*n)
        p = self.p

        # compute mean
        if self.whiten:
            curr_mean = 0
            for el in data:
                curr_mean += el.shape[0] * np.mean(el, axis=0)
            self.mean = curr_mean / sum(len_samples)

            # mean shift
            for i in range(len(data)):
                data[i] = data[i] - self.mean

        if verbose:
            print("computing SVD decomposition.")

        if svd_algo == 'cols':
            V, s, Uh = Svd_single_column(data, self.p, k=approx_k, verbose=verbose)
            s = np.diag(s).copy()
        elif svd_algo == 'exact':
            data = build_xhi_matrix(data, verbose)
            V, s, Uh = la.svd(data, full_matrices=False)
        elif svd_algo == 'fb_pca':
            data = build_xhi_matrix(data, verbose)
            V, s, Uh = fbpca.pca(data, k=p, raw=True, n_iter=5)
        elif svd_algo == 'sparse_exact':
            dim_x = data[0].shape[1]
            t_max = max(len_samples)
            t_sum = sum(len_samples)
            linop = splinalg.LinearOperator((t_sum, t_max * dim_x),
                                            matvec=lambda v: xi_data_matvec(data, v),
                                            rmatvec=lambda v: xi_data_rmatvec(data, v))
            V, s, Uh = splinalg.svds(linop, k=self.p)
        else:
            assert False

        V = V[:, :p]
        s = s[:p]
        Uh = Uh[:p, :]

        inv_s = s.copy()
        inv_s[s > self.epsilon] = 1 / s[s > self.epsilon]
        vtrv = vt_R_v_block_multiplication(V, len_samples)
        Q = np.diag(inv_s) @ vtrv @ np.diag(s)

        self.A = Uh.T[:k, :].T
        self.B = Q
        self.sigma = s

    def encode(self, x, save_history=False):
        """
        Args:
            x: array-like (batch, time, features)

        Returns: last hidden state for each sequence (batch, p)
        """
        assert len(x.shape) == 3
        x = x - self.mean
        n = x.shape[1]
        if self.A is None:
            raise NotFittedError()

        y0 = np.zeros(self.B.shape[0])
        hist = []
        for i in range(n):
            y0 = x[:, i, :] @ self.A.T + y0 @ self.B.T
            if save_history:
                hist.append(y0)

        if save_history:
            return np.stack(hist, axis=1)
        else:
            return y0

    def decode(self, y, n):
        assert len(y.shape) == 2
        if self.A is None:
            raise NotFittedError()

        reco = []
        for i in range(n):
            reco.append(np.asarray(y @ self.A + self.mean))
            y = y @ self.B
        reco = np.stack(reco)
        return reco[::-1].transpose([1, 0, 2])

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            d = pickle.load(f)
        self.__dict__.update(d)
        return self
