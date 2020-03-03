import os
import pickle

import torch

from .utils import cuda_move
import logging
import torch.nn as nn
from logging import Logger
from .callbacks import EarlyStoppingCallback, LearningCurveCallback, ModelCheckpoint
from tqdm import tqdm
import json


def build_default_logger(log_dir: str, debug=False) -> Logger:
    log_mode = logging.INFO
    if debug:
        log_mode = logging.DEBUG
    default_logger: Logger = logging.getLogger(log_dir + '_' + 'log_output')
    default_logger.setLevel(log_mode)

    # log to output file
    log_file = log_dir + 'output.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_mode)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    default_logger.addHandler(fh)
    # log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(log_mode)
    ch.setFormatter(formatter)
    default_logger.addHandler(ch)
    return default_logger


class TorchTrainer:
    def __init__(self, model: nn.Module, n_epochs: int=100, log_dir: str=None, callbacks=None, patience: int=5000,
                 verbose=True, logger=None, validation_steps=1, checkpoint_mode='loss', debug=False):
        assert checkpoint_mode in {'loss', 'metric', None}
        self.checkpoint_mode = checkpoint_mode
        self.debug = debug
        if checkpoint_mode == 'loss':
            self.is_improved_performance = TorchTrainer._is_improved_loss
        elif checkpoint_mode == 'metric':
            self.is_improved_performance = TorchTrainer._is_improved_metric
        elif checkpoint_mode is None:
            self.is_improved_performance = lambda x: True
        else:
            assert False
        self.model = cuda_move(model)
        self.n_epochs = n_epochs
        self.log_dir = log_dir
        self.verbose = verbose
        self.callbacks = [] if callbacks is None else callbacks
        self.callbacks.extend([
            EarlyStoppingCallback(patience),
            LearningCurveCallback()
        ])
        if checkpoint_mode is not None:
            self.callbacks.append(ModelCheckpoint(log_dir))
        self._init_fit_history()
        self.logger = logger
        self.validation_steps = validation_steps
        self._stop_train = False
        os.makedirs(log_dir, exist_ok=True)
        self._hyperparams_dict = {
            'n_epochs': self.n_epochs,
            'callbacks': self.callbacks,
        }
        if logger is None:
            self.logger = build_default_logger(log_dir, debug)

    def _init_fit_history(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_result = {}
        self.best_vl_metric = -1e15
        self.best_vl_loss = 1e9
        self.best_loss_epoch = -1  # used by _is_improved_loss when best_vl_loss is already updated
        self.best_metric_epoch = -1  # used by _is_improved_metric when best_vl_metric is already updated
        self.best_epoch = 0
        self.global_step = 0
        self._stop_train = False
        self.best_result = {
            'tr_loss': 10 ** 10,
            'tr_acc': -10 ** 10,
            'vl_loss': 10 ** 10,
            'vl_acc': -10 ** 10,
            'best_epoch': -1
        }

    def append_hyperparam_dict(self, d):
        self._hyperparams_dict = {**self._hyperparams_dict, **d}

    def validate(self, train_data, val_data):
        e = self.global_step
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

    def resume_fit(self, train_data, validation_data):
        # load last model
        device = f'cuda:{torch.cuda.current_device()}'
        self.model = torch.load(self.log_dir + 'model_e.pt', device)

        # load fit history
        # init training variables
        with open(self.log_dir + 'checkpoint.pickle', 'rb') as f:
            d = pickle.load(f)

        self.train_losses = d['tr_loss']
        self.val_losses = d['vl_loss']
        self.train_metrics = d['tr_accs']
        self.val_metrics = d['vl_accs']

        self.best_result = d['best_result']
        self.best_vl_metric = d['best_result']['vl_acc']
        self.best_epoch = None

        self.global_step = len(self.train_losses)
        self._stop_train = False

        # call _fit
        self._fit(train_data, validation_data)

    def fit(self, train_data, validation_data):
        self._init_fit_history()
        self._init_training(train_data, validation_data)
        self.logger.info("Starting training...")
        self.logger.info("Params: {}".format(self._hyperparams_dict))
        for cb in self.callbacks:
            cb.before_training(self)

        try:
            self._fit(train_data, validation_data)  # training loop
        except KeyboardInterrupt:
            self.logger.info("Training stopped due to keyboard interrupt.")
            for cb in self.callbacks:
                cb.after_training_interrupted(self)
            raise KeyboardInterrupt()

    def _fit(self, train_data, validation_data):
        for e in range(self.global_step, self.n_epochs):
            self.global_step = e
            self.fit_epoch(train_data)

            if e % self.validation_steps == 0:
                self.validate(train_data, validation_data)
                self.save_checkpoint(e)

            for cb in self.callbacks:
                cb.after_epoch(self, train_data, validation_data)

            if self._stop_train:
                self.logger.info(f"Training stopped at epoch {e + 1}")
                break

        for cb in self.callbacks:
            cb.after_train_before_validate(self)
        self.logger.info("Final Training Results:")
        self.validate(train_data, validation_data)
        self.save_checkpoint(e)
        for cb in self.callbacks:
            cb.after_training(self)
        return self.train_losses, self.val_losses

    def stop_train(self):
        self._stop_train = True

    def _is_improved_metric(self):
        return self.best_vl_metric < self.val_metrics[-1] or self.best_loss_epoch == self.global_step

    def _is_improved_loss(self):
        return self.best_vl_loss > self.val_losses[-1] or self.best_metric_epoch == self.global_step

    def save_checkpoint(self, e):
        if self.is_improved_performance(self):
            self.logger.debug('updating best result.')
            self.best_epoch = e
            self.best_result = {
                'tr_loss': self.train_losses[-1],
                'tr_acc': self.train_metrics[-1],
                'vl_loss': self.val_losses[-1],
                'vl_acc': self.val_metrics[-1],
                'best_epoch': self.best_epoch
            }
        if self._is_improved_loss():
            self.logger.debug('improved loss.')
            self.best_vl_loss = self.val_losses[-1]
            self.best_loss_epoch = self.global_step
        if self._is_improved_metric():
            self.logger.debug('improved metric.')
            self.best_vl_metric = self.val_metrics[-1]
            self.best_metric_epoch = self.global_step

        d = {
            'train_params': self._hyperparams_dict,
            'best_result': self.best_result,
            'tr_loss': self.train_losses,
            'vl_loss': self.val_losses,
            'tr_accs': self.train_metrics,
            'vl_accs': self.val_metrics
        }
        # save JSON checkpoint
        with open(self.log_dir + 'checkpoint.json', 'w') as f:
            json.dump(d, f, indent=4, default=lambda x: str(x))

    def __str__(self):
        s = self.__class__.__name__
        return s

    def train_dict(self):
        raise DeprecationWarning()

    def compute_metrics(self, data):
        raise NotImplementedError

    def fit_epoch(self, train_data):
        raise NotImplementedError

    def _init_training(self, train_data, val_data):
        """ Initialize optimizers and other variables useful during the training. """
        # raise NotImplementedError
        # TODO: deprecate? with callbacks and the optimizer argument it is probably useless.
        pass


class SequentialTaskTrainer(TorchTrainer):
    def __init__(self, model, optimizer, n_epochs=100, log_dir=None, regularizers=None, callbacks=None, grad_clip=10, patience=5000, **kwargs):
        super().__init__(model, n_epochs, log_dir, patience=patience, **kwargs)
        self.opt = optimizer
        self.grad_clip = grad_clip
        self.append_hyperparam_dict({
            'optimizer': optimizer,
            'grad_clip': self.grad_clip
        })
        self.regularizers = [] if regularizers is None else regularizers
        if callbacks:
            self.callbacks.extend(callbacks)

    def compute_metrics(self, data):
        with torch.no_grad():
            self.model.eval()
            err = 0
            acc = 0
            bi = 0
            for xi, yi in tqdm(data.iter()):
                assert len(xi.shape) == 3
                y_pred = self.model(xi)
                err += xi.shape[1] * data.loss_score(y_pred, yi)
                acc += xi.shape[1] * data.metric_score(y_pred, yi)
                bi += xi.shape[1]
            err = err / bi
            acc = acc / bi

            if torch.isnan(err):
                self.logger.info("NaN loss. Stopping training...")
                self.stop_train()

        return float(err), float(acc)

    def fit_epoch(self, train_data):
        self.model.train()
        for xi, yi in tqdm(train_data.iter()):
            y_pred = self.model(xi)
            err = train_data.loss_score(y_pred, yi)

            reg = 0.0
            for reg_foo in self.regularizers:
                reg += reg_foo(self.model, xi, yi)
            err = err + reg

            self.model.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.opt.step()

            for cb in self.callbacks:
                cb.after_backward(xi)


class GenericTaskTrainer(TorchTrainer):
    def __init__(self, model, optimizer, n_epochs=100, log_dir=None, regularizers=None, **kwargs):
        super().__init__(model, n_epochs, log_dir, **kwargs)
        self.opt = optimizer
        self.append_hyperparam_dict({
            'optimizer': optimizer
        })
        self.regularizers = [] if regularizers is None else regularizers

    def compute_metrics(self, data):
        with torch.no_grad():
            self.model.eval()
            err = 0
            acc = 0
            bi = 0
            for batch in tqdm(data.iter()):
                y_pred = self.model(batch)
                err += data.loss_score(y_pred, batch).detach()
                acc += data.metric_score(y_pred, batch).detach()
                bi += 1
            err = err / bi
            acc = acc / bi

            if torch.isnan(err):
                self.logger.info("NaN loss. Stopping training...")
                self.stop_train()

        return float(err), float(acc)

    def fit_epoch(self, train_data):
        self.model.train()
        for batch in tqdm(train_data.iter()):
            y_pred = self.model(batch)
            err = train_data.loss_score(y_pred, batch)

            reg = 0.0
            for reg_foo in self.regularizers:
                reg += reg_foo(self.model, batch)
            err = err + reg

            self.model.zero_grad()
            err.backward()
            self.opt.step()
