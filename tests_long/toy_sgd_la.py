from sgd_la import *
import cannon.laes.svd_la as cla
import numpy as np


def reco_error(model, X):
    t_max = X.shape[0]
    H = []

    ht = model.init_hidden(X.shape[1])
    for t in range(t_max):
        xt = X[t]
        ht = model(xt, ht)
        H.append(ht)

    H = torch.stack(H, dim=0)

    ht = H[-1]
    x_reco = []
    for t in range(t_max):
        ht, x_delay_t = model.decode(ht)
        x_reco.append(x_delay_t)

    x_reco = torch.stack(x_reco[::-1], dim=0)
    return torch.mean((X - x_reco) ** 2)


def complete_reco_error(model, X):
    t_max = X.shape[0]
    H = []

    ht = model.init_hidden(X.shape[1])
    for t in range(t_max):
        xt = X[t]
        ht = model(xt, ht)
        H.append(ht)

    H = torch.stack(H, dim=0)

    err = 0
    sum_coeff = 0
    for k in range(H.shape[0]):
        ht = H[k]
        x_reco = []
        for t in range(k + 1):
            ht, x_delay_t = model.decode(ht)
            x_reco.append(x_delay_t)

        x_reco = torch.stack(x_reco[::-1], dim=0)
        curr_mse = torch.mean((X[:k + 1] - x_reco) ** 2)
        err += k * curr_mse
        sum_coeff += k
    return err / sum_coeff


if __name__ == '__main__':
    n_samples = 64
    n_time = 20
    n_features = 31
    hidden_size = 100
    lambda_reg = 1
    train_data = torch.randn(n_time, n_samples, n_features)


    n_epochs = 500
    model = SequentialLinearAutoencoderCell(n_features, hidden_size)
    opt = torch.optim.Adam(model.parameters())
    for ne in range(n_epochs):
        mse_err = complete_reco_error(model, train_data)
        reg = soft_sla_ortho_constraints(model)
        loss = mse_err + lambda_reg * reg
        model.zero_grad()
        loss.backward()
        opt.step()
        if ne % 100 == 99:
            print("epoch {} reconstruction error: {}".format(ne + 1, mse_err))


    train_data = train_data.numpy().transpose(1, 0, 2)
    svd_model = cla.LinearAutoencoder(hidden_size)
    svd_model.fit(train_data)

    yn = svd_model.encode(train_data)
    reco = svd_model.decode(yn, n_time)
    svd_mse = np.mean((train_data - reco) ** 2)
    print("SVD LA reconstruction error: {}".format(svd_mse))
    print("Done.")
