from cannon.utils import set_allow_cuda
from cannon.rnn import LMNLayer
from cannon.container import DiscreteRNN
import torch


def test_rnn():
    set_allow_cuda(False)
    K = 10
    rnn = LMNLayer(100, 100, 100)
    model = DiscreteRNN(K, K, 100, rnn=rnn)

    fake_input = torch.zeros(20, K).long()
    model(fake_input)

    f_model = './tests_long/experiment_repo/try_save.ptj'
    model.save(f_model)
    model = torch.jit.load(f_model)
    model(fake_input)
    print("Done.")
