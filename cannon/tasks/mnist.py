from cannon.tasks import Dataset
import torch.nn.functional as F
from torchvision import datasets
import torch
from cannon.utils import cuda_move
from cannon.container import DiscreteRNN


class MNISTDigit(Dataset):
    def __init__(self, data_dir, set_key='train', debug=False, batch_size=64):
        """
        MNIST Digit Dataset.

        Args:
            data_dir: file directory
            set_key: one of 'train', 'valid', 'test'
            debug: if True only loads a small number of samples (e.g. 100)
        """
        super().__init__()
        self.set_key = set_key
        self.debug = debug
        self.batch_size = batch_size
        bool_train = set_key != 'test'
        raw_data = datasets.MNIST(data_dir, train=bool_train, download=True)

        if set_key == 'train':
            self.data = raw_data.train_data[:50000]
            self.labels = raw_data.train_labels[:50000]
        elif set_key == 'valid':
            self.data = raw_data.train_data[50000:]
            self.labels = raw_data.train_labels[50000:]
        elif set_key == 'test':
            self.data = raw_data.test_data
            self.labels = raw_data.test_labels

        if debug:
            self.data = self.data[:1000]
            self.labels = self.labels[:1000]
        self.data = cuda_move(self.data.float())
        self.labels = cuda_move(self.labels)

    def iter(self):
        n_samples = self.data.shape[0]
        if self.set_key == 'train':
            # shuffle dataset
            idxs = torch.randperm(n_samples)
            self.data = self.data[idxs]
            self.labels = self.labels[idxs]
        for ii in range(0, n_samples, self.batch_size):
            xi = self.data[ii: ii+self.batch_size]
            yi = self.labels[ii: ii+self.batch_size]
            yield xi / 255.0, yi

    @property
    def input_shape(self):
        return 1, 28, 28

    @property
    def output_shape(self):
        return 1, 10

    def metric_score(self, y_pred, y_target):
        """ Accuracy score. """
        assert len(y_pred.shape) == 2
        return (y_pred.argmax(dim=1) == y_target).float().mean()

    def loss_score(self, y_pred, y_target):
        """ Cross-entropy loss. """
        return F.cross_entropy(y_pred, y_target)

    def visualize_sample(self, x, y):
        """ Single sample visualization. Batch dimension must be removed. """
        x = x.reshape(28, 28)
        str_rows = []
        for i in range(28):
            sr = "".join(str((el > 0.2).item()) for el in x[i])
            sr = sr.translate(str.maketrans({'0': ' ', '1': 'x'}))
            str_rows.append(sr)
        str_img = "\n".join(str(row) for row in str_rows)
        str_x = " img:\n" + str_img
        str_y = "class:" + str(y.argmax(dim=0).item())
        return str_x + '\n' + str_y


class SequentialPixelMNIST(MNISTDigit):
    def __init__(self, data_dir, set_key='train', debug=False, batch_size=64):
        """
        MNIST Digit Dataset.

        Args:
            data_dir: file directory
            set_key: one of 'train', 'valid', 'test'
            debug: if True only loads a small number of samples (e.g. 100)
        """
        super().__init__(data_dir, set_key, debug, batch_size)

    def iter(self):
        for xi, yi in super().iter():
            xi = xi.reshape(-1, 28*28)\
                .transpose(0, 1)\
                .unsqueeze(2)
            yield xi, yi

    @property
    def input_shape(self):
        return 28 * 28, 1, 1

    def visualize_sample(self, x, y):
        return super().visualize_sample(x.reshape(28, 28), y)


class PermutedPixelMNIST(MNISTDigit):
    def __init__(self, data_dir, set_key='train', debug=False, batch_size=64, perm_file=None):
        """
        MNIST Digit Dataset.

        Args:
            data_dir: file directory
            set_key: one of 'train', 'valid', 'test'
            debug: if True only loads a small number of samples (e.g. 100)
        """
        super().__init__(data_dir, set_key, debug, batch_size)
        if perm_file:
            self.permute = torch.load(perm_file)
        else:
            self.permute = torch.randperm(28*28)

    def iter(self):
        for xi, yi in super().iter():
            xi = xi.reshape(-1, 28*28)
            xi = xi[:, self.permute]
            xi = xi.transpose(0, 1).unsqueeze(2)
            yield xi, yi

    @property
    def input_shape(self):
        return 28 * 28, 1, 1

    def visualize_sample(self, x, y):
        """ Don't visualize permuted samples. """
        return None


def test_mnist():
    data = MNISTDigit('~/data/mnist', set_key='train', debug=True)
    print(data.visualize_sample(data.data[0], data.labels[0]))
    for _ in data.iter():
        pass

    data = SequentialPixelMNIST('~/data/mnist', set_key='train', debug=True)
    print(data.visualize_sample(data.data[0], data.labels[0]))
    for x, y in data.iter():
        assert x.shape[0] == 28*28

    data = PermutedPixelMNIST('~/data/mnist', set_key='train', debug=True)
    print(data.visualize_sample(data.data[0], data.labels[0]))
    for x, y in data.iter():
        assert x.shape[0] == 28 * 28


if __name__ == '__main__':
    test_mnist()
    print("Done.")
