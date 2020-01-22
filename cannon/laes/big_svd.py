"""
Large scale SVD. Use a column-partitioned computation.
Works with dense matrices.

original implementation: https://pdfs.semanticscholar.org/49b1/9f7884f625a0ff0f59deb7d11635481da6e0.pdf?_ga=2.186766120.1275162375.1542634957-367157004.1513342723
"""
import numpy as np
import scipy.sparse as sparse
from scipy.stats import ortho_group
import numpy.linalg as la
import math


def get_Xi_block(data, n_block):
    len_samples = [len(el) for el in data]
    tot_len = sum(len_samples)
    Xhii = np.zeros((tot_len, data[0][0].shape[0]), dtype=np.float32)

    sum_prev_samples = 0
    for sample_i, sample in enumerate(data):
        for t_step in range(len(sample) - n_block):
            Xhii[sum_prev_samples + t_step + n_block, :] = sample[t_step]
        sum_prev_samples += len_samples[sample_i]
    return Xhii


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
        for jj in range(num_slices):
            xii = get_Xi_block(data, jj)
            curr_slice.append(xii)
        curr_slice = np.hstack(curr_slice)
        V, s, Uh = la.svd(curr_slice)
        return V[:,:n_components], np.diag(s[:n_components]), Uh[:n_components,:]

    last_slice = get_Xi_block(data, num_slices - 1)
    # compute svd for the last slice, and then repeat this process for each slice concatenated with the previous result.
    v, s, u_t = KeCSVD(last_slice, n_components)
    del last_slice
    for i in reversed(range(0, num_slices - 1, k)):
        if verbose:
            print("slice", i, " of ", num_slices)
        curr_vs = v @ s
        del v, s, u_t

        curr_slice = []
        for jj in range(i, min(i + k, num_slices - 1)):
            xii = get_Xi_block(data, jj)
            curr_slice.append(xii)
        curr_slice = np.hstack(curr_slice)

        curr_vs = np.hstack((curr_slice, curr_vs))
        v, s, u_t = KeCSVD(curr_vs, n_components)
        del curr_vs
    return v, s, u_t


def SvdForBigData(data, sample_dim, n_components, k=1, verbose=False):
    """ Compute the SVD for big matrices, approximating the result.

    Args:
        data: input matrix.
        sample_dim: dimension of a single slice.
        n_components: number of principal components to return as output.
    """
    sample_dim = k * sample_dim
    num_slices = int(math.ceil(data.shape[1] / (sample_dim)))
    idx_start = int(np.ceil(n_components / sample_dim))
    last_slice = data[:, (num_slices - idx_start) * sample_dim:]

    # compute svd for the last slice, and then repeat this process for each slice concatenated with the previous result.
    v, s, u_t = KeCSVD(last_slice, n_components)
    num_slices = num_slices - idx_start
    for i in reversed(range(num_slices)):
        if verbose:
            print("slice", i, " of ", num_slices)
        curr_vs = v @ s
        curr_slice = data[:, i * sample_dim:(i + 1) * sample_dim]
        curr_vs = np.hstack((curr_slice, curr_vs))
        v, s, u_t = KeCSVD(curr_vs, n_components)
    return v, s, u_t


def indirectSVD(M, p):
    """
    SVD decomposition.

    Args:
        M: sparse input matrix
        p: number of principal components.
    """
    Q, R = np.linalg.qr(M)
    v_r, s, u_t = np.linalg.svd(R)
    v = Q.dot(v_r)

    s = s[0:p]
    v = v[:, 0:p]
    u_t = u_t[0:p, :]

    s = sparse.csc_matrix((s.tolist(), (range(s.shape[0]), range(s.shape[0]))), shape=(v.shape[1], u_t.shape[0]))
    s = s.todense()
    return v, s, u_t


def KeCSVD(M, p=10, min_sigma=0.000001):
    c = 0
    if M.shape[0] <= M.shape[1]:
        # svd on kernel matrix
        Kernel = M @ M.transpose()
        v, Ssqr, _ = indirectSVD(Kernel, p)
        s = np.sqrt(Ssqr)
        u_t = np.linalg.pinv(s) @ v.transpose() @ M
    else:
        # svd on convariance matrix
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


if __name__ == "__main__":
    n = 100
    p = 100
    m = np.random.rand(n, n)
    (v, s, u_t) = SvdForBigData(m, p + 1, p + 1)
    # print(v)
    vv, _, _ = np.linalg.svd(m)
    # print(v - vv[:, :p])

    mv = ortho_group.rvs(dim=n)
    mu = ortho_group.rvs(dim=n)
    s = np.random.randn(n)
    s[p:] = 0
    s = np.diag(s)
    A = mv @ s @ mu

    a, b, c = SvdForBigData(A, p + 1, p + 1)
    c = c[:, :p]
    assert la.norm(A[:, :p] - (a @ b @ c)) / la.norm(A[:, :p]) < 0.01
