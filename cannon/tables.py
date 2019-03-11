import os
import pickle


def load_dir_results(log_dir):
    print(f"Reporting results in {log_dir}")
    res = []
    for file in os.scandir(log_dir):
        if os.path.isdir(file):
            log_file = log_dir + file.name + '/checkpoint.pickle'

            with open(log_file, 'rb') as f:
                d = pickle.load(f)
                best_result = d['best_result']
                train_par = d['train_params']
                model_par = d['model_params']

            res.append((best_result, train_par, model_par))
    return res