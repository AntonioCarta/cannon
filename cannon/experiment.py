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
        os.makedirs(log_dir, exist_ok=True)
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

    def run(self, config=None):
        """
        Launch the experiment with a given configuration.
        :param config: Experiment configuration
        """
        self.experiment_log.info("Starting new run.")
        # self.experiment_log.info("Configuration: " + config.pretty_print())
        self.foo(config)
        self.experiment_log.info("Experiment terminated.")



    def foo(self, config=None):
        raise NotImplementedError()


class Config:
    def pretty_print(self, d=None):
        if d is None:
            return json.dumps(self.__dict__)
        else:
            return str(d)

