import numpy as np
import scipy.linalg as la
from scipy import sparse
from sklearn.exceptions import NotFittedError
from cannon.la.skl import svd_sign_flip
from .big_svd import Svd_single_column, SvdForBigData
import pickle


def build_xhi_matrix(data):
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
        reversed_sample = sample[::-1, :].reshape(-1)
        for t_step in range(len(sample)):
            row_idx = sum_prev_samples + t_step
            Xhi[row_idx, :t_step*n_features + n_features] = reversed_sample[-t_step*n_features - n_features:]
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

    def fit(self, data, approximate=False, approx_k=1, verbose=False):
        """
        Fit the Linear Autoencoder using SVD-based training.

        Args:
            data: list of sequences (possibly with different lengths)
        """
        if self.A is not None:
            print("linear autoencoder has been already trained.")

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
        if approximate:
            V, s, Uh = Svd_single_column(data, self.p, k=approx_k, verbose=verbose)
            # data = build_xhi_matrix(data)
            # V, s, Uh = SvdForBigData(data, approx_k, p, verbose=verbose)
            s = np.diag(s).copy()
        else:
            data = build_xhi_matrix(data)
            V, s, Uh = la.svd(data, full_matrices=False)

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
            return np.vstack(hist)
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


class IncrementalLinearAutoencoder(LinearAutoencoder):
    def __init__(self, p, forget_factor=1.0, epsilon=1e-7):
        super().__init__(p, epsilon=epsilon)
        # parameters
        self.tot_samples = 0
        self.forget_factor = forget_factor
        self.u = None
        self.s = None
        self.vt = None
        self.utRu = None
        self.mean = 0

    def partial_fit(self, data):
        """
        Fit the Linear Autoencoder using incremental SVD-based training.
        NOTE: due to implementation issues sentences must have the same length.

        Args:
            data: numpy array of shape (batch, time, features) or list of arrays (time, features).
        """
        len_samples = [el.shape[0] for el in data]
        batch_size = len(len_samples)
        t = max(len_samples)
        k = data[0].shape[-1]
        self.p = min(self.p, k*t, batch_size*t)
        p = self.p
        Xhi = build_xhi_matrix(data)
        if self.A is not None:
            # sn = self.tot_samples
            # sm = sum(len_samples)
            # # compute mean
            # curr_mean = 0
            # for ii in range(data.shape[0]):
            #     el = data[ii]
            #     curr_mean += el.shape[0] * np.mean(el, axis=0)
            # curr_mean = curr_mean / sm
            #
            # # mean shift
            # for i in range(len(data)):
            #     data[i] = data[i] - curr_mean
            #
            # # update mean
            # self.mean = sm / (sn + sm) * curr_mean + sn / (sn + sm) * self.mean

            f_vt_s = self.forget_factor * self.vt.T * self.s.reshape(1, -1)
            cat_usb = np.concatenate([f_vt_s, Xhi.T], axis=1)
            L, X = la.qr(cat_usb, mode='full')
            v_bar, s_bar, u_bar_t = la.svd(X, full_matrices=False)
            v_bar = v_bar[:, :p]
            s_bar = s_bar[:p]
            u_bar_t = u_bar_t[:p, :]
            a1, a2 = svd_sign_flip(u_bar_t.T, v_bar.T)
            u_bar_t, v_bar = a1.T, a2.T

            u_correction = np.zeros((self.u.shape[0] + Xhi.shape[0], self.u.shape[1] + Xhi.shape[0]))
            u_correction[:self.u.shape[0], :self.u.shape[1]] = self.u
            idxs_row = range(self.u.shape[0], self.u.shape[0] + Xhi.shape[0])
            idxs_col = range(self.u.shape[1], self.u.shape[1] + Xhi.shape[0])
            u_correction[idxs_row, idxs_col] = 1

            u = u_correction @ u_bar_t.T[:, :self.p]
            s = s_bar[:self.p]
            vt = v_bar.T[:self.p, :] @ L.T

            s[s < self.epsilon] = 0
            inv_s = s.copy()
            inv_s[s > self.epsilon] = 1 / s[s > self.epsilon]

            ll = sum(len_samples)
            Rho = np.zeros((p + ll, p + ll))
            R = np.asarray(build_R(len_samples).todense())
            Rho[:p, :p] = self.utRu
            Rho[p:, p:] = R.T

            Q = np.diag(s) @ u_bar_t @ Rho @ u_bar_t.T @ np.diag(inv_s)
            utRu = u_bar_t @ Rho @ u_bar_t.T
        else:
            # sm = sum(len_samples)
            # # compute mean
            # curr_mean = 0
            # for el in data:
            #     curr_mean += el.shape[0] * np.mean(el, axis=0)
            #
            # # update mean
            # self.mean = curr_mean / sm
            #
            # # mean shift
            # for i in range(len(data)):
            #     data[i] = data[i] - self.mean

            u, s, vt = la.svd(Xhi, full_matrices=False)
            u = u[:, :p]
            s = s[:p]
            vt = vt[:p, :]
            u, vt = svd_sign_flip(u, vt)

            R = build_R(len_samples)
            R = np.asarray(R.todense())

            s[s < self.epsilon] = 0
            inv_s = s.copy()
            inv_s[s > self.epsilon] = 1 / s[s > self.epsilon]
            Q = np.diag(s) @ u.T @ R.T @ u @ np.diag(inv_s)
            utRu = np.asarray(u.T @ R.T @ u)

        self.tot_samples += sum(len_samples)
        self.u, self.s, self.vt = u, s, vt
        self.utRu = utRu
        self.A = vt.T[:k, :].T
        self.B = Q.T

    def get_memory_states(self, x):
        assert len(x.shape) == 3
        x = x - self.mean
        n = x.shape[1]
        if self.A is None:
            raise NotFittedError()

        ys = []
        y0 = np.zeros(self.p)
        for i in range(n):
            y0 = x[:, i, :] @ self.A.T + y0 @ self.B.T
            ys.append(y0)
        return np.stack(ys).transpose([1, 0, 2])

    def decode_step(self, y):
        assert len(y.shape) == 3  # (batch, time, features)
        if self.A is None:
            raise NotFittedError()

        y_prev = np.zeros_like(y)
        x_prev = []
        for ti in range(y.shape[1]):
            yi = y[:, ti, :]
            x_prev.append(self.A.T @ yi.T + self.mean)
            y_prev[:, ti, :] = yi @ self.B
        return np.stack(x_prev).transpose([2, 0, 1]), y_prev
