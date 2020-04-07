import torch
import numpy as np
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cannon.utils import cuda_move
from cannon.tasks import Dataset
from cannon.callbacks import TrainingCallback
import matplotlib.pyplot as plt


class PlotGenSequenceCB(TrainingCallback):
    def __init__(self, log_dir, data, plot_every=1):
        super().__init__()
        self.log_dir = log_dir
        self.plot_every = plot_every
        self.data = data

    def after_epoch(self, model_trainer, train_data, validation_data):
        if (model_trainer.global_step + 1) % self.plot_every == 0:
            model_trainer.logger.debug("PlotGenSequenceCB: plotting generated sequence.")
            self.plot_generated_sequence(model_trainer)

    def after_training(self, model_trainer):
        self.plot_generated_sequence(model_trainer)

    def plot_generated_sequence(self, model_trainer):
        for x, y in self.data.iter():
            y_pred = model_trainer.model(x).detach().cpu().numpy().ravel()

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(y.detach().cpu().numpy().ravel(), label='true')
        ax.plot(y_pred, label='gen')
        ax.set_title(f'generated sequence (epoch={model_trainer.global_step})')
        ax.set_xlabel("t")
        ax.set_xlabel("y")
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure('fig')
        fig.savefig(self.log_dir + 'gen_seq.pdf')
        plt.close(fig)


class SequenceGeneration(Dataset):
    def __init__(self, data_dir, seq_id):
        """
            Sequence Generation tasks. The model does not receive any input and it must return a desired sequence as output.
            ref: Clockwork RNN
        """
        super().__init__()
        self.data_dir = data_dir
        self.seq_id = seq_id
        self.ys = np.load(data_dir + 'samples.npy')[seq_id].reshape(-1, 1, 1)
        self.ys = cuda_move(torch.tensor(self.ys).float())

    def iter(self):
        # x is just a dummy input because this dataset does not have an input
        x = cuda_move(torch.zeros(self.ys.shape[0], 1, 1))
        yield x, self.ys

    def loss_score(self, batch, y_pred):
        x, y = batch
        return F.mse_loss(y_pred, y)

    def __str__(self):
        return f"SequenceGeneration Dataset (dat_dir={self.data_dir}, seq_id={self.seq_id})"


if __name__ == '__main__':
    data_dir = '/home/carta/data/wavegen/'
    data = SequenceGeneration(data_dir, 0)

    for s in data.iter():
        assert len(s[1].shape) == 3
        assert s[1].shape[1] == 1

        y_rand = torch.randn_like(data.ys)
        data.loss_score(s, y_rand)
        data.metric_score(s, y_rand)

    print("Done.")
