import os
import pickle

import torch

from .experiment import Experiment, Config
from .utils import cuda_move
import logging
import torch.nn as nn
from collections import namedtuple
from itertools import product
from logging import Logger


class LMTRainingConfig(Config):
    def __init__(self, param_list, train_data, val_data, batch_size, n_epochs, patience=50):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.param_list = param_list
        self.train_data = train_data
        self.val_data = val_data
        self.patience = patience

    def pretty_print(self, d=None):
        d = self.__dict__.copy()
        d['train_data'] = str(self.train_data)
        d['val_data'] = str(self.val_data)
        return str(d)


SerializableParameter = namedtuple('SerializableParameter', ['name', 'value'])
TrainerConfig = namedtuple('TrainerConfig', ['batch_size', 'n_epochs', 'callbacks', 'patience', 'l2_loss', 'verbose'])


class PLTConfig(Config):
    def __init__(self, model_params, train_params, train_data, val_data):
        self.model_params = model_params
        self.train_params = train_params
        self.train_data = train_data
        self.val_data = val_data

    def pretty_print(self, d=None):
        d = self.__dict__.copy()
        d['train_data'] = str(self.train_data)
        d['val_data'] = str(self.val_data)
        return str(d)


def build_default_logger(log_dir : str) -> Logger:
    default_logger : Logger = logging.getLogger(log_dir + '_' + 'log_output')
    default_logger.setLevel(logging.DEBUG)

    # log to output file
    log_file = log_dir + 'output.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    default_logger.addHandler(fh)
    # log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    default_logger.addHandler(ch)
    return default_logger


def json_save_dict(obj_dict, file_name):
    def json_save_fp(d, fp, indent=0):
        ind = '\t' * indent
        fp.write('{\n')
        for k, v in d.items():
            if type(v) is dict:
                fp.write(ind + '\t"{}": '.format(k))
                json_save_fp(v, fp, indent + 1)
            elif type(v) is list:
                fp.write(ind + '\t"{}": [\n'.format(k))
                for el in v:
                    fp.write(ind + '\t\t{},\n'.format(el))
                fp.write(ind + '\t]\n')
            else:
                fp.write(ind + '\t"{}": {},\n'.format(k, v))
        fp.write(ind + '},\n')

    with open(file_name, 'w') as f:
        json_save_fp(obj_dict, f)


class ParamListTrainer(Experiment):
    def __init__(self, log_dir, model_class, trainer_class, resume_ok=True):
        self.results = []
        super().__init__(log_dir, resume_ok=resume_ok)
        self._model_class = model_class
        self._trainer_class = trainer_class

    def save_checkpoint(self, config):
        # TODO: correct resume to working with model_pars x train_pars product
        d = {'results': self.results, 'param_list': config.model_params}
        with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
            pickle.dump(d, f)
        json_save_dict(d, self.log_dir + 'checkpoint.json')

    def foo(self, config: PLTConfig):
        self.results = []
        self.experiment_log.info("Starting to train with {} different configurations.".format(len(config.model_params)))
        self.experiment_log.info("{}".format(config.train_params))

        start_i = 0
        if self.resume_ok:
            self.load_checkpoint()
            if len(self.results) > 0:
                self.experiment_log.info("Resuming experiment from configuration {}.".format(len(self.results)))
                start_i = len(self.results)

        model_train_prod = list(product(config.model_params, config.train_params))
        for i, (model_par, train_par) in enumerate(model_train_prod[start_i:]):
            self.experiment_log.info("Model parameters: {}".format(model_par))
            self.experiment_log.info("Train parameters: {}".format(train_par))
            train_log_dir = self.log_dir + 'k_{}/'.format(i)
            os.makedirs(train_log_dir, exist_ok=True)

            model = self._model_class(**model_par)
            trainer = self._trainer_class(model=model, **train_par, log_dir=train_log_dir)
            # try:
            #     model_trainer.fit(config.train_data, config.val_data)
            #     res = model_trainer.best_result
            # except Exception as e:
            #     self.experiment_log.error("training configuration failed with error: {}".format(e))
            #     res = { 'tr_loss': np.NaN, 'tr_acc': np.NaN, 'vl_loss': np.NaN, 'vl_acc': np.NaN }
            trainer.fit(config.train_data, config.val_data)
            res = trainer.best_result

            self.results.append(res)
            self.save_checkpoint(config)
            self.experiment_log.info("TR loss: {}, metric: {}".format(res['tr_loss'], res['tr_acc']))
            self.experiment_log.info("VL loss: {}, metric: {}".format(res['vl_loss'], res['vl_acc']))

    def load_checkpoint(self):
        if os.path.exists(self.log_dir + 'checkpoint.pickle'):
            with open(self.log_dir + 'checkpoint.pickle', 'rb') as f:
                d = pickle.load(f)
                if 'results' in d:
                    self.results = d['results']


class TorchTrainer:
    def __init__(self, model: nn.Module, n_epochs: int=100, log_dir: str=None,
                 callbacks=None, patience: int=50, verbose=True, logger=None, validation_steps=1):
        self.model = cuda_move(model)
        self.n_epochs = n_epochs
        self.log_dir = log_dir
        self.verbose = verbose
        self.patience = patience
        self.callbacks = [] if callbacks is None else callbacks
        self._init_fit_history()
        self.logger = logger
        self.serializable_params = []
        self.validation_steps = validation_steps
        os.makedirs(log_dir, exist_ok=True)
        if logger is None:
            self.logger = build_default_logger(log_dir)

    def add_serializable_parameters(self, params):
        self.serializable_params.extend(params)

    def _init_fit_history(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_result = {}
        self.best_vl_metric = -1e15
        self.best_epoch = 0
        self.global_step = -1

    def train_dict(self):
        return {
            'n_epochs': self.n_epochs,
            'patience': self.patience,
            'callbacks': self.callbacks
        }

    def validate(self, e, train_data, val_data):
        me, acc = self.compute_metrics(train_data)
        self.train_losses.append(me)
        self.train_metrics.append(acc)
        if self.verbose:
            self.logger.info("epoch %d TRAIN loss: %f" % (e + 1, me))
            self.logger.info("epoch %d TRAIN metric: %f" % (e + 1, acc))
        me, acc = self.compute_metrics(val_data)
        self.val_losses.append(me)
        self.val_metrics.append(acc)
        if self.verbose:
            self.logger.info("epoch %d VAL loss: %f" % (e + 1, me))
            self.logger.info("epoch %d VAL metric: %f\n" % (e + 1, acc))

    def fit(self, train_data, validation_data):
        self._init_fit_history()
        self._init_training(train_data, validation_data)
        self.logger.info("Starting training...")
        self.logger.info("Params: {}".format(str(self)))

        for cb in self.callbacks:
            cb.before_training(self)

        for e in range(self.n_epochs):
            self.global_step = e
            self.fit_epoch(train_data)

            if e % self.validation_steps == 0:
                self.validate(e, train_data, validation_data)
                self.save_checkpoint(e)

            for cb in self.callbacks:
                cb.after_epoch(self, train_data, validation_data)

            if self.best_vl_metric >= self.val_metrics[-1] and e - self.best_epoch > self.patience:
                # print("self.best_vl_metric({}) >= self.val_metrics[-1]({}) and e({}) - self.best_epoch({})> self.patience({})"
                #       .format(self.best_vl_metric, self.val_metrics[-1], e, self.best_epoch, self.patience))
                self.logger.info("Early stopping at epoch {}".format(e))
                break

        self.validate(e, train_data, validation_data)
        self.save_checkpoint(e)
        for cb in self.callbacks:
            cb.after_training(self)
        return self.train_losses, self.val_losses

    def save_checkpoint(self, e):
        torch.save(self.model, self.log_dir + 'model_e.pt')
        if self.best_vl_metric < self.val_metrics[-1]:
            self.best_result = {
                'tr_loss': self.train_losses[-1],
                'tr_acc': self.train_metrics[-1],
                'vl_loss': self.val_losses[-1],
                'vl_acc': self.val_metrics[-1]
            }
            self.best_vl_metric = self.val_metrics[-1]
            self.best_epoch = e
            torch.save(self.model, self.log_dir + 'best_model.pt')

        train_params = self.train_dict()
        for p in self.serializable_params:
            train_params[p.name] = p.value
        d = {
            'model_params': self.model.params_dict(),
            'train_params': train_params,
            'best_result': self.best_result,
            'tr_loss': self.train_losses,
            'vl_loss': self.val_losses,
            'tr_accs': self.train_metrics,
            'vl_accs': self.val_metrics
        }

        # save pickle checkpoint (if possible)
        try:
            with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
                pickle.dump(d, f)
        except:
            pass

        # save JSON checkpoint
        json_save_dict(d, self.log_dir + 'checkpoint.json')

    def __str__(self):
        s = self.__class__.__name__ + ': '
        # s += json.dumps(self.__dict__)
        for par in self.serializable_params:
            s += par.name + ' ' + str(par.value) + ', '
        return s

    def compute_metrics(self, data):
        raise NotImplementedError

    def fit_epoch(self, train_data):
        raise NotImplementedError

    def _init_training(self, train_data, val_data):
        """ Initialize optimizers and other variables useful during the training. """
        raise NotImplementedError
