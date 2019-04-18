"""
    Classes and utilities for model selection.
"""
import random
from cannon.experiment import Experiment
import pickle
from cannon.torch_model import json_save_dict
import os
from itertools import product
import json


class RandomSampler:
    def __init__(self, param_dists):
        self.param_dists = param_dists

    def sample(self):
        new_config = {}
        for name, dist in self.param_dists.items():
            if type(dist) == list:
                new_config[name] = random.choice(dist)
            else:
                new_config[name] = dist.rvs()
        return new_config


class ParamListTrainer(Experiment):
    def __init__(self, log_dir, param_list, train_foo, resume_ok=True):
        self.results = []
        super().__init__(log_dir, resume_ok=resume_ok)
        self.param_list = param_list
        self.train_foo = train_foo
        self.results = []

    def save_checkpoint(self, config):
        # TODO: correct resume to working with model_pars x train_pars product
        d = {'results': self.results, 'param_list': self.param_list}
        with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
            pickle.dump(d, f)
        # json_save_dict(d, self.log_dir + 'checkpoint.json')
        with open(self.log_dir + 'checkpoint.json', 'w') as f:
            json.dump(d, f, indent=4, default=lambda x: str(x))

    def foo(self, config=None):
        self.results = []
        start_i = 0
        if self.resume_ok:
            self.load_checkpoint()
            if len(self.results) > 0:
                self.experiment_log.info("Resuming experiment from configuration {}.".format(len(self.results)))
                start_i = len(self.results)

        self.experiment_log.info("Starting to train with {} different configurations.".format(len(self.param_list[start_i:])))
        for pl in self.param_list[start_i:]:
            self.experiment_log.info("{}".format(f'\t{pl}'))

        for i, params in enumerate(self.param_list[start_i:]):
            self.experiment_log.info("Configuration: {}".format(params))
            train_log_dir = self.log_dir + 'k_{}/'.format(i)
            os.makedirs(train_log_dir, exist_ok=True)

            res = self.train_foo(log_dir=train_log_dir, params=params)
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
                    self.param_list = d['param_list']
