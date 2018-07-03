import json
import logging
import os
import pickle
import sys

import numpy as np


class Experiment:
    def __init__(self, log_dir, resume_ok=True):
        """
        Base class for all experiments. Responsible for the logger configuration and to launch the main task.
        :param log_dir: directory to save the experiment's output.
        :param resume_ok: if True an partially completed experiment can be resumed.
        """
        self.log_dir = log_dir
        self.resume_ok = resume_ok

        try:
            os.makedirs(self.log_dir, exist_ok=resume_ok)
        except OSError:
            print("{} directory already exists. Try with a different name.".format(self.log_dir))
            sys.exit(0)

        self.experiment_log = logging.getLogger(self.log_dir)
        self.experiment_log.setLevel(logging.DEBUG)
        log_file = self.log_dir + 'logbook.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S')
        fh.setFormatter(formatter)
        self.experiment_log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.experiment_log.addHandler(ch)

        self.experiment_log.info('-' * 10)
        self.experiment_log.info("Experiment started.")

    def run(self, config):
        """
        Launch the experiment with a given configuration.
        :param config: Experiment configuration
        """
        self.experiment_log.info("Starting new run.")
        self.experiment_log.info("Configuration: " + config.pretty_print())
        self.foo(config)
        self.experiment_log.info("Experiment terminated.")

    def foo(self, config):
        raise NotImplementedError()


class Config:
    def pretty_print(self, d=None):
        if d is None:
            return json.dumps(self.__dict__)
        else:
            return str(d)


class ModelSelectionConfig(Config):
    def __init__(self, model_builder, train_data, val_data, num_gpus, trials, k=3):
        """
        ModelSelection experimental settings.
        :param model_builder: object used to create the models
        :param train_data: training dataset with Dataset API
        :param val_data: validation dataset with Dataset API
        :param num_gpus: number of GPU used in parallel
        :param trials: number of models to train
        :param k: number of trials for each selected configuration
        """
        self.model_builder = model_builder
        self.train_data = train_data
        self.val_data = val_data
        self.num_gpus = num_gpus
        self.trials = trials
        self.k = k

    def pretty_print(self, d=None):
        d = self.__dict__.copy()
        d['model_builder'] = self.model_builder.__class__.__name__
        d['train_data'] = str(self.train_data)
        d['val_data'] = str(self.val_data)
        return str(d)


class ModelSelection(Experiment):
    """
    Experiment to perform the model selection.
    """
    def __init__(self, log_dir, resume_ok=True):
        super().__init__(log_dir, resume_ok=resume_ok)

    def load_checkpoint(self):
        if os.path.exists(self.log_dir + 'checkpoint.pickle'):
            with open(self.log_dir + 'checkpoint.pickle', 'rb') as f:
                d = pickle.load(f)
            return d['remaining_ids'], d['results_mapping'], d['parameters_mapping']
        else:
            return None, None, None

    def save_checkpoint(self, remaining_ids, results_mapping, parameters_mapping):
        d = {
            'remaining_ids': remaining_ids,
            'results_mapping': results_mapping,
            'parameters_mapping': parameters_mapping
        }
        with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
            pickle.dump(d, f)

    def foo(self, config):
        remaining_ids = None
        if self.resume_ok:
            remaining_ids, results_mapping, parameters_mapping = self.load_checkpoint()
            if remaining_ids is not None:
                if len(remaining_ids) > 0:
                    self.experiment_log.info("resuming from iteration {}".format(remaining_ids[0]))
                else:
                    self.experiment_log.info("model selection already terminated.")

        if remaining_ids is None:
            remaining_ids = [i for i in range(config.trials)]
            results_mapping = {}
            parameters_mapping = {}

        for i in remaining_ids:
            p = config.model_builder.generate_params(i)
            parameters_mapping[i] = p

            train_exp = TrainingExperiment(self.log_dir + 'config_{}/'.format(i), self.resume_ok)
            train_config = TrainingConfig(config.model_builder, p, config.train_data, config.val_data, config.k)
            train_exp.run(train_config)
            results = train_exp.results
            results_mapping[i] = results

            error = results['vl_loss_avg']        # VL_AVG_LOSS on K-folds
            self.experiment_log.info("Finished iteration {}".format(i))
            self.experiment_log.info("VAL loss: {}".format(error))
            self.experiment_log.info("Model params: {}".format(p))

            self.experiment_log.info("Saving checkpoint.\n")
            self.save_checkpoint([x for x in range(i + 1, config.trials)], results_mapping, parameters_mapping)

        # look for best config
        best_error = np.inf
        best_parameters = {}
        best_results = {}
        for k in parameters_mapping.keys():
            error = results_mapping[k]['vl_loss_avg']
            if error < best_error:
                best_parameters = parameters_mapping[k]
                best_error = error
                best_results = results_mapping[k]

        # Record the best performing set of parameters.
        self.save_checkpoint([x for x in range(i + 1, config.trials)], results_mapping, parameters_mapping)
        self.experiment_log.info("Best VAL loss over {} trials was {}".format(config.trials, best_error))
        self.experiment_log.info("Best VAL acc over {} trials was {}".format(config.trials, best_results['vl_acc_avg']))
        self.experiment_log.info("Model params: {}\n".format(best_parameters))
        self.best_parameters = best_parameters
        self.best_results = best_results


class TrainingConfig(Config):
    def __init__(self, model_builder, params, train_data, val_data, k):
        self.model_builder = model_builder
        self.params = params
        self.train_data = train_data
        self.val_data = val_data
        self.k = k

    def pretty_print(self, d=None):
        d = self.__dict__.copy()
        d['model_builder'] = self.model_builder.__class__.__name__
        d['train_data'] = str(self.train_data)
        d['val_data'] = str(self.val_data)
        return str(d)


class TrainingExperiment(Experiment):
    def __init__(self, log_dir, resume_ok=True):
        super().__init__(log_dir, resume_ok=resume_ok)

    def save_checkpoint(self, params, tr_err, vl_err, tr_acc, vl_acc):
        d = {
            'params': params,
            'tr_err': tr_err,
            'vl_err': vl_err,
            'tr_acc': tr_acc,
            'vl_acc': vl_acc,
            'results': self.results
        }
        with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
            pickle.dump(d, f)

    def foo(self, config):
        self.experiment_log.info("Model parameters: {}".format(config.params))
        tr_err, vl_err, tr_acc, vl_acc = [], [], [], []
        for x in range(config.k):
            os.makedirs(self.log_dir + 'k_{}/'.format(x), exist_ok=True)
            config.model_builder.log_dir = self.log_dir + 'k_{}/'.format(x)
            model = config.model_builder.build_model(config.params)
            # Train
            hist = model.fit(config.train_data, config.val_data)
            res = model.best_result
            tr_err.append(res['tr_loss'])
            vl_err.append(res['vl_loss'])
            tr_acc.append(res['tr_acc'])
            vl_acc.append(res['vl_acc'])

        self.results = {
            'tr_loss_avg': np.average(tr_err),
            'tr_loss_std': np.std(tr_err),
            'vl_loss_avg': np.average(vl_err),
            'vl_loss_std': np.std(vl_err),

            'tr_acc_avg': np.average(tr_acc),
            'tr_acc_std': np.std(tr_acc),
            'vl_acc_avg': np.average(vl_acc),
            'vl_acc_std': np.std(vl_acc)
        }

        self.save_checkpoint(config.params, tr_err, vl_err, tr_acc, vl_acc)
        self.experiment_log.info("TR error avg {}, std {}".format(np.average(tr_err), np.std(tr_err)))
        self.experiment_log.info("VL error avg {}, std {}".format(np.average(vl_err), np.std(vl_err)))
        self.experiment_log.info("VL acc   avg {}, std {}".format(np.average(vl_acc), np.std(vl_acc)))