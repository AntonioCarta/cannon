class LMNPretrainingRNN(TrainingCallback):
    def __init__(self, train_data, val_data, la_file):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.la_file = la_file
        self.la = LinearAutoencoder(300)
        self.la.load(la_file)

    def _encode_data(self, data):
        X, H, y = [], [], []
        for xi, (yi, ti) in data.iter():
            assert len(xi.shape) == 3
            assert xi.shape[1] == 1
            X.append(xi[:, 0, :].cpu().numpy())
            y.append(yi.cpu().numpy())
        for xi in X:
            hh = self.la.encode(xi[None, :, :], save_history=True)
            H.append(hh)
        H = np.concatenate(H, axis=0)
        y = np.concatenate(y)
        return H, y

    def before_training(self, model_trainer):
        model = model_trainer.model.rnn.layer

        model_trainer.logger.info("Training linear regressor.")
        h_train, y = self._encode_data(self.train_data)
        y_onehot = np.eye(61)[y[:, 0]]
        W, _, _, _ = np.linalg.lstsq(h_train, y_onehot)

        y_pred = h_train @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_tr_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        h_val, y = self._encode_data(self.val_data)
        y_pred = h_val @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_vl_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        model_trainer.logger.info("Initializing matrices.")
        model.Wxh.data = torch.tensor(self.la.A).float()
        model.bh.data = torch.zeros_like(model.bh.data)
        model.Wmh.data = torch.zeros_like(model.Wmh.data)
        model.Whm.data = torch.eye(model.memory_size).float()
        model.Wmm.data = torch.tensor(self.la.B).float()
        model.bm.data = torch.tensor(-(self.la.A @ self.la.mean).reshape(-1)).float()
        model_trainer.model.ro.weight.data = torch.tensor(torch.tensor(W.T)).float()
        model_trainer.model.ro.bias.data = torch.tensor(torch.zeros_like(model_trainer.model.ro.bias)).float()
        cuda_move(model_trainer.model)

        tr_err, tr_acc = model_trainer.compute_metrics(self.train_data)
        vl_err, vl_acc = model_trainer.compute_metrics(self.val_data)
        model_trainer.logger.info("**************************************************************")
        model_trainer.logger.info(f"* Loaded checkpoint from: {self.la_file}")
        model_trainer.logger.info(f" Linear RNN: TRAIN acc {la_tr_acc}")
        model_trainer.logger.info(f" Linear RNN init: VALID acc {la_vl_acc}")
        model_trainer.logger.info(f" After model init: TRAIN loss {tr_err}, acc {tr_acc}")
        model_trainer.logger.info(f" After model init: VALID loss {vl_err}, acc {vl_acc}")
        model_trainer.logger.info("**************************************************************")

    def __str__(self):
        return self.__class__.__name__ + '(TrainingCallback)'


class RNNPretraining(TrainingCallback):
    def __init__(self, train_data, val_data, la_file):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.la_file = la_file
        self.la = LinearAutoencoder(300)
        self.la.load(la_file)

    def _encode_data(self, data):
        X, H, y = [], [], []
        for xi, (yi, ti) in data.iter():
            assert len(xi.shape) == 3
            assert xi.shape[1] == 1
            X.append(xi[:, 0, :].cpu().numpy())
            y.append(yi.cpu().numpy())
        for xi in X:
            hh = self.la.encode(xi[None, :, :], save_history=True)
            H.append(hh)
        H = np.concatenate(H, axis=0)
        y = np.concatenate(y)
        return H, y

    def before_training(self, model_trainer):
        model = model_trainer.model.rnn.layer

        model_trainer.logger.info("Training linear regressor.")
        h_train, y = self._encode_data(self.train_data)

        # readout = LogisticRegressionCV(cv=1, n_jobs=10)
        # readout.fit(h_train, y)
        y_onehot = np.eye(61)[y[:, 0]]
        W, _, _, _ = np.linalg.lstsq(h_train, y_onehot)

        # y_pred = readout.predict(h_train)
        # la_tr_acc = (y_pred == y.reshape(-1)).mean()
        y_pred = h_train @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_tr_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        # h_val, y = self._encode_data(self.val_data)
        # y_pred = readout.predict(h_val)
        # la_vl_acc = (y_pred == y.reshape(-1)).mean()
        h_val, y = self._encode_data(self.val_data)
        y_pred = h_val @ W
        y_pred = np.argmax(y_pred, axis=1)
        la_vl_acc = (y_pred.reshape(-1) == y.reshape(-1)).sum() / y.shape[0]

        model_trainer.logger.info("Initializing matrices.")
        model.Wxh.data = torch.tensor(self.la.A).float()
        model.Whh.data = torch.tensor(self.la.B).float()
        model.bh.data = torch.tensor(-(self.la.A @ self.la.mean).reshape(-1)).float()
        # model_trainer.model.ro.weight.data = torch.tensor(readout.coef_).float()
        # model_trainer.model.ro.bias.data = torch.tensor(readout.intercept_).float()
        model_trainer.model.ro.weight.data = torch.tensor(W.T).float()
        model_trainer.model.ro.bias.data = torch.zeros_like(model_trainer.model.ro.bias)

        cuda_move(model_trainer.model)
        tr_err, tr_acc = model_trainer.compute_metrics(self.train_data)
        vl_err, vl_acc = model_trainer.compute_metrics(self.val_data)
        model_trainer.logger.info("**************************************************************")
        model_trainer.logger.info(f"* Loaded checkpoint from: {self.la_file}")
        model_trainer.logger.info(f" Linear RNN: TRAIN acc {la_tr_acc}")
        model_trainer.logger.info(f" Linear RNN init: VALID acc {la_vl_acc}")
        model_trainer.logger.info(f" After model init: TRAIN loss {tr_err}, acc {tr_acc}")
        model_trainer.logger.info(f" After model init: VALID loss {vl_err}, acc {vl_acc}")
        model_trainer.logger.info("**************************************************************")

    def __str__(self):
        return self.__class__.__name__ + '(TrainingCallback)'
