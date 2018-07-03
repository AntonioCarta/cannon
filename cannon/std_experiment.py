import os
import pickle
import torch
import numpy as np
from .experiment import Config, Experiment


class TKConfig(Config):
    def __init__(self, model_params, train_params, train_data, val_data, test_data):
        self.model_params = model_params
        self.train_params = train_params
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def pretty_print(self, d=None):
        d = self.__dict__.copy()
        d['train_data'] = str(self.train_data)
        d['val_data'] = str(self.val_data)
        d['test_data'] = str(self.test_data)
        return str(d)


class TrainK(Experiment):
    def __init__(self, log_dir, model_class, k, resume_ok=True):
        self.results = []
        self.k = k
        super().__init__(log_dir, resume_ok=resume_ok)
        self._model_class = model_class

    def save_checkpoint(self, config):
        # TODO: correct resume to working with model_pars x train_pars product
        d = {'results': self.results, 'model_params': config.model_params, 'train_params': config.train_params}
        with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
            pickle.dump(d, f)

    def foo(self, config: TKConfig):
        self.results = []
        self.experiment_log.info("Starting to train for {} iterations.".format(self.k))
        self.experiment_log.info("{}".format(config.train_params))

        train_accs, val_accs, test_accs = [], [], []
        for i in range(self.k):
            train_log_dir = self.log_dir + 'k_{}/'.format(i)
            os.makedirs(train_log_dir, exist_ok=True)

            model_trainer = self._model_class(config.model_params, **config.train_params, log_dir=train_log_dir)
            try:
                model_trainer.fit(config.train_data, config.val_data)
                res = model_trainer.best_result
            except Exception as e:
                self.experiment_log.error("training configuration failed with error: {}".format(e))
                res = { 'tr_loss': np.NaN, 'tr_acc': np.NaN, 'vl_loss': np.NaN, 'vl_acc': np.NaN }

            self.results.append(res)
            self.save_checkpoint(config)
            self.experiment_log.info("TR error {}, ACC {}".format(res['tr_loss'], res['tr_acc']))
            self.experiment_log.info("VL error {}, ACC {}".format(res['vl_loss'], res['vl_acc']))

            model_trainer.model = torch.load(train_log_dir + 'best_model.pt')
            train_accs.append(model_trainer.compute_metrics(config.train_data)[1])
            val_accs.append(model_trainer.compute_metrics(config.val_data)[1])
            test_accs.append(model_trainer.compute_metrics(config.test_data)[1])

        self.experiment_log.info("Best results:")
        self.experiment_log.info("TRAIN mean {}, std {}".format(np.mean(train_accs), np.std(train_accs)))
        self.experiment_log.info("VALID mean {}, std {}".format(np.mean(val_accs), np.std(val_accs)))
        self.experiment_log.info("TEST  mean {}, std {}".format(np.mean(test_accs), np.std(test_accs)))

    def load_checkpoint(self):
        if os.path.exists(self.log_dir + 'checkpoint.pickle'):
            with open(self.log_dir + 'checkpoint.pickle', 'rb') as f:
                d = pickle.load(f)
                if 'results' in d:
                    self.results = d['results']
