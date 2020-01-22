import os
from cannon.torch_trainer import TorchTrainer
from cannon.utils import cuda_move, set_gpu
import torch
from torch import nn

set_gpu()


class MockTrainer(TorchTrainer):
    def __init__(self, model):
        super().__init__(model, log_dir='./debug_logs/')

    def compute_metrics(self, data):
        return -self.global_step, self.global_step

    def fit_epoch(self, train_data):
        if (self.global_step + 1) % 5 == 0:
            self.stop_train()

    def _init_training(self, train_data, val_data):
        pass

    def train_dict(self):
        return {}


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = cuda_move(torch.zeros(5,5))

    def params_dict(self):
        return {'a': 0}


def test_resume_fit():
    import gpustat

    is_first = True
    for i in range(4):
        stats = gpustat.GPUStatCollection.new_query()
        ratios = list(map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats))
        if ratios[i] > 0.8:
            print(f"Full memory. Skipping GPU {i}")
            continue  # full memory, try next gpu

        print(f"Setting to GPU {i}")
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        model = MockModel()
        trainer = MockTrainer(model)

        if is_first:
            is_first = False
            trainer.fit(None, None)
        else:
            trainer.resume_fit(None, None)


if __name__ == '__main__':
    test_resume_fit()