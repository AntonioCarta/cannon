import torch
import torch.nn.functional as F

from mslmn import MultiScaleLMN
from clockwork_lmn.incremental_train_new import IncrementalTrainingCallback
from cannon.tasks import Dataset
from cannon.utils import cuda_move
from cannon.torch_trainer import build_default_logger
from clockwork_lmn.container import ItemClassifier


class DummyData(Dataset):
    def __init__(self, x_shape, y_shape):
        """ Dataset with random data """
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape

    def iter(self):
        x = cuda_move(torch.randn(*self.x_shape))
        y = cuda_move(torch.randn(*self.y_shape))

        t_x = torch.tensor([self.x_shape[0] for _ in range(x.shape[1])])
        t_y = torch.tensor([self.y_shape[0] for _ in range(x.shape[1])])
        yield x, (y, t_x, t_y)

    def loss_score(self, batch, y_pred):
        return F.mse_loss(y_pred, batch[1])


class DummyTrainer:
    def __init__(self, model):
        self.model = model
        self.logger = build_default_logger(log_dir, debug=True)

    def compute_metrics(self, data):
        return -1, -1


if __name__ == '__main__':
    ms = 13
    params = {'pretrain_every': 2, 'memory_size': ms}
    log_dir = './logs/debug/'

    x_shape = (65, 32, 11)
    y_shape = x_shape
    data = DummyData(x_shape, y_shape)

    rnn = MultiScaleLMN(x_shape[2], 5, ms, num_modules=1, max_modules=5)
    cb = IncrementalTrainingCallback(params, [rnn], data, data, data, log_dir)

    model = cuda_move(ItemClassifier(rnn, ms*5, 7))
    trainer = DummyTrainer(model)

    for x, y in data.iter():
        break

    y_old = model(x)
    cb.pretrain_module(trainer)
    y_new = model(x)
    assert ((y_old - y_new) ** 2).sum() == 0

    cb.pretrain_module(trainer)
    m_prev = rnn.init_hidden(x.shape[1])
    y_new = model(x)
    assert ((y_old - y_new) ** 2).sum() == 0

    print("Done")
