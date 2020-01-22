import numpy as np
import scipy.linalg as la
from sklearn.decomposition import IncrementalPCA

from cannon.laes.skl import incremental_svd_cols, incremental_svd_rows, svd_sign_flip


def print_abs_error(u1, s1, v1, u2, s2, v2):
    u_error = la.norm(np.abs(u1) - np.abs(u2))
    s_error = la.norm(np.abs(s1) - np.abs(s2))
    vt_error = la.norm(np.abs(v1) - np.abs(v2))
    print("\tabs errors: U: {:3f}, S: {:3f}, V^T: {:3f}".format(u_error, s_error, vt_error))


def test_flip_sign_equivalence():
    n, m, k = 100, 200, 20
    A = np.random.randn(n, m)
    u1, s1, vt1 = la.svd(A)
    u1 = u1[:, :k]
    s1 = s1[:k]
    vt1 = vt1[:k, :]

    u2, vt2 = svd_sign_flip(u1, vt1)
    s2 = s1

    A1 = (u1 * s1[np.newaxis, :]) @ vt1
    A2 = (u2 * s2[np.newaxis, :]) @ vt2
    np.testing.assert_allclose(A1, A2)


def test_svd():
    print("Incremental SVD test")
    d = 25  # number of features
    n = 7000  # number of samples
    m = 300   # number of new samples
    k = 12

    A = np.random.randn(d, n)
    B = np.random.randn(d, m)
    cat_ab = np.concatenate((A, B), axis=1)

    A = A - np.mean(A, axis=1)[:, np.newaxis]
    B = B - np.mean(B, axis=1)[:, np.newaxis]

    u_A, s_A, vt_A = la.svd(A)
    u_A = u_A[:, :k]
    s_A = s_A[:k]
    vt_A = vt_A[:k, :]
    u_cols, s_cols, vt_cols = incremental_svd_cols(u_A, s_A, vt_A, B, k)
    u_cols, vt_cols = svd_sign_flip(u_cols, vt_cols)

    u_svd, s_svd, vt_svd = la.svd(cat_ab)
    u_svd = u_svd[:, :k]
    s_svd = s_svd[:k]
    vt_svd = vt_svd[:k, :]
    u_svd, vt_svd = svd_sign_flip(u_svd, vt_svd)

    u_error = la.norm(u_svd - u_cols)
    s_error = la.norm(s_svd - s_cols)
    vt_error = la.norm(vt_svd - vt_cols)
    print("\t(columns) --> errors: U: {:3f}, S: {:3f}, V^T: {:3f}".format(u_error, s_error, vt_error))
    print_abs_error(u_svd, s_svd, vt_svd, u_cols, s_cols, vt_cols)

    u_svd, s_svd, vt_svd = la.svd(cat_ab.T)
    u_svd = u_svd[:, :k]
    s_svd = s_svd[:k]
    vt_svd = vt_svd[:k, :]
    vt_svd, u_svd = svd_sign_flip(vt_svd.T, u_svd.T)
    vt_svd, u_svd = vt_svd.T, u_svd.T

    u_rows, s_rows, vt_rows = incremental_svd_rows(vt_A.T, s_A, u_A.T, B.T, k)
    vt_rows, u_rows= svd_sign_flip(vt_rows.T, u_rows.T)
    vt_rows, u_rows = vt_rows.T, u_rows.T

    u_error = la.norm(u_svd - u_rows)
    s_error = la.norm(s_svd - s_rows)
    vt_error = la.norm(vt_svd - vt_rows)
    print("\t(rows) --> errors: U: {:3f}, S: {:3f}, V^T: {:3f}".format(u_error, s_error, vt_error))
    print_abs_error(u_svd, s_svd, vt_svd, u_rows, s_rows, vt_rows)

    np.testing.assert_allclose(u_cols, vt_rows.T)
    np.testing.assert_allclose(vt_cols, u_rows.T)

    iPCA = IncrementalPCA(n_components=k)
    iPCA.partial_fit(A.T)
    iPCA.partial_fit(B.T)

    u_ipca = iPCA.components_.T
    u_ipca = svd_sign_flip(u_ipca)
    s_ipca = iPCA.singular_values_
    print("\t(sklearn) u_pca error: {:3f}, s_pca error: {:3f}".format(la.norm(vt_svd.T - u_ipca), la.norm(s_svd - s_ipca)))
    u_error = la.norm(np.abs(vt_svd.T) - np.abs(u_ipca))
    print("\t(sklearn) abs errors: U: {:3f}".format(u_error))


if __name__ == '__main__':
    test_flip_sign_equivalence()
    print("test_flip_sign_equivalence: OK.")
    test_svd()
    print("test_svd: OK.")
