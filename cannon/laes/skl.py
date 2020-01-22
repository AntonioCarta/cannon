import numpy as np
import scipy.linalg as la


def incremental_svd_cols(u, s, vt, B, k):
    s = np.diag(s)
    cat_usb = np.concatenate((u @ s, B), axis=1)
    Q, R = la.qr(cat_usb, mode='full')
    u_bar, s_bar, vt_bar = la.svd(R)
    u_bar = u_bar[:, :k]
    s_bar = s_bar[:k]
    vt_bar = vt_bar[:k, :]

    v_correction = np.zeros((vt.shape[0] + B.shape[1], vt.shape[1] + B.shape[1]))
    v_correction[:vt.shape[0], :vt.shape[1]] = vt
    idxs_row = range(vt.shape[0], vt.shape[0] + B.shape[1])
    idxs_col = range(vt.shape[1], vt.shape[1] + B.shape[1])
    v_correction[idxs_row, idxs_col] = 1

    u_res = Q @ u_bar
    s_res = s_bar
    vt_res = vt_bar @ v_correction

    u_res, vt_res = svd_sign_flip(u_res, vt_res)
    return u_res, s_res, vt_res


def svd_sign_flip(u, v=None):
    """ Solve sign indeterminacy for SVD decomposition.
        from sklearn/utils/extmath.py/_deterministic_vector_sign_flip
    """
    if v is not None:
        max_abs_rows = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_rows, range(u.shape[1])])
        v = v * signs[:, np.newaxis]
        u = u * signs
        return u, v
    else:
        max_abs_rows = np.argmax(np.abs(u), axis=1)
        signs = np.sign(u[range(u.shape[0]), max_abs_rows])
        u = u * signs[:, np.newaxis]
        return u


def incremental_svd_rows(u, s, vt, B, k):
    cat_usb = np.concatenate([vt.T * s.reshape(1, -1), B.T], axis=1)
    Q, R = la.qr(cat_usb, mode='full')
    v_bar, s_bar, u_bar_t = la.svd(R)

    u_correction = np.zeros((u.shape[0] + B.shape[0], u.shape[1] + B.shape[0]))
    u_correction[:u.shape[0], :u.shape[1]] = u
    idxs_row = range(u .shape[0], u.shape[0] + B.shape[0])
    idxs_col = range(u.shape[1], u.shape[1] + B.shape[0])
    u_correction[idxs_row, idxs_col] = 1

    u_res = u_correction @ u_bar_t.T[:, :k]
    s_res = s_bar[:k]
    vt_res = v_bar.T[:k, :] @ Q.T

    u_res, vt_res = svd_sign_flip(u_res, vt_res)
    return u_res, s_res, vt_res
