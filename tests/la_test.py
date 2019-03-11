import numpy as np
import scipy.linalg as la
from cannon.tasks.dataset import mackey_glass
from cannon.la.svd_la import LinearAutoencoder, IncrementalLinearAutoencoder, build_xhi_matrix
from cannon.la.skl import svd_sign_flip
from cannon.tasks import PianoRollData


def _create_data(batch_size, k, n):
    data = mackey_glass(sample_len=batch_size*k*n, tau=17)[0]
    data = np.reshape(data, (batch_size, k, n))
    data = np.transpose(data, [0, 2, 1])
    #data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data


def test_la_piano_midi():
    f_data = '../../data/midi/jsb_chorales.pickle'
    data = PianoRollData(f_data, key='train')
    t_max = 20
    n = 80

    X = []
    for i in range(len(data)):
        el = data[i]
        if el[0].shape[0] >= t_max:
            X.append(el[0][:t_max].numpy())
    X = np.stack(X, 0)[:n]

    lina = IncrementalLinearAutoencoder(p=800)
    for i in range(0, n, 40):
        lina.partial_fit(X[i: i + 50])

    ys = lina.encode(X)
    reco = lina.decode(ys, t_max)
    reco_err = la.norm(X - reco) / la.norm(X)
    print("JSB Chorales (subset) mean reco. error: {}".format(reco_err))
    assert reco_err < 0.01


def test_la():
    n = 1000
    data = _create_data(batch_size=3, k=2, n=n) + 3
    lin_ae = LinearAutoencoder(p=2000)
    lin_ae.fit(data)

    y = lin_ae.encode(data)
    reco = lin_ae.decode(y, n)
    print("LA error: {}".format(la.norm(data - reco) / la.norm(data)))
    np.testing.assert_allclose(data, reco, rtol=1.e-5)


def test_inc_la():
    n=100
    data = _create_data(batch_size=30, k=10, n=n) + 5
    b_cut = 10

    inc_ae = IncrementalLinearAutoencoder(p=2000, forget_factor=1.0)
    inc_ae.partial_fit(data[:b_cut])
    inc_ae.partial_fit(data[b_cut:])
    y = inc_ae.encode(data)
    reco = inc_ae.decode(y, n)
    print("IncrementalLA reconstruction error: {:5e}".format(la.norm(data - reco) / la.norm(data)))
    assert la.norm(data - reco) / la.norm(data) < 0.0001

    Xhi = build_xhi_matrix(data)
    u, s, vt = la.svd(Xhi, full_matrices=False)
    u = u[:, :2000]
    s = s[:2000]
    vt = vt[:2000, :]
    u, vt = svd_sign_flip(u, vt)
    inc_ae.u, inc_ae.vt = svd_sign_flip(inc_ae.u, inc_ae.vt)
    u_err = la.norm(u - inc_ae.u) / la.norm(u)
    s_err = la.norm(s - inc_ae.s) / la.norm(s)
    vt_err = la.norm(vt - inc_ae.vt) / la.norm(vt)
    print("IncLA SVD errors: U {:3e}, S {:3e}, Vt {:3e}".format(u_err, s_err, vt_err))
    assert u_err < 0.0001
    assert s_err < 0.0001
    assert vt_err < 0.0001


def test_inc_la_forget_factor():
    n=100
    data = _create_data(batch_size=30, k=10, n=n)
    b_cut = 10

    inc_ae = IncrementalLinearAutoencoder(p=2000, forget_factor=0.5)
    inc_ae.partial_fit(data[:b_cut])
    inc_ae.partial_fit(data[b_cut:])
    y = inc_ae.encode(data)
    reco = inc_ae.decode(y, n)
    print("IncrementalLA - forget_factor=0.5 reconstruction error: {:5e}".format(la.norm(data - reco) / la.norm(data)))
    assert la.norm(data - reco) / la.norm(data) < 0.0001


if __name__ == '__main__':
    test_la()
    test_inc_la()
    test_inc_la_forget_factor()
    test_la_piano_midi()