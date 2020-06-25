"""
    Common interface for datasets (used at least for most of the synthetic tasks in cannon).
"""
import torch.nn.functional as F


class Dataset:
    def __init__(self):
        pass

    @property
    def input_shape(self):
        """ Input shape. If batch is not set by the class, it is set to 1. """
        raise NotImplementedError()

    @property
    def output_shape(self):
        """ Output shape. If batch is not set by the class, it is set to 1. """
        raise NotImplementedError()

    def iter(self):
        """ Dataset iterator.

            Returns: tuple (x, y). If some dataset need more than one input (e.g. masks) it must return it as a tuple
                (x, x_mask), (y, y_mask).
        """
        raise NotImplementedError()

    def metric_score(self, batch, y_pred):
        """ Compute the metric given a predicted and target sample. By default the metric is just the opposite
            of the loss.
         """
        return -self.loss_score(batch, y_pred)

    def loss_score(self, batch, y_pred):
        """ Compute the loss given a predicted and target sample. """
        raise NotImplementedError()

    def visualize_sample(self, x, y):
        """ Single sample visualization. Batch dimension must be removed. """
        print("Sample visualization not implemented for the current class.")


class AutoencodingData(Dataset):
    def __init__(self, data):
        """
        A generic class to transform a dataset into an input autoencoding problem.

        Args:
            data:
        """
        super().__init__()
        self.data = data

    def iter(self):
        for batch in self.data.iter():
            yield batch

    def loss_score(self, batch, y_pred):
        return F.mse_loss(batch[0], y_pred)