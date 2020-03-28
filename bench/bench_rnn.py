"""
    benchmark script for RNN models implemented using torch jit.

    Improved performance of nn.Module alternatives by using Pytorch JIT.
    NOTE: serialization may not work for some models.

    Currently Jit models are about 3x faster.
    CuDNN LSTM is still 3x faster than a comparable LMN.

    JitLMN min: 0.00875, mean: 0.00896, n_trials: 10
    SlowLMN min: 0.02461, mean: 0.03130, n_trials: 10
    SlowLSTM min: 0.00317, mean: 0.00329, n_trials: 10
"""
from cannon.utils import set_allow_cuda, set_gpu, cuda_move, timeit
from cannon.rnn_jit import RNNLayer, LMNLayer
from cannon.container import DiscreteRNN
import torch
import torch
from torch import nn
import torch.nn.functional as F
from cannon.utils import cuda_move
from torch import jit
from cannon.rnn_jit import LinearMemoryNetwork
from torch.jit import Tensor


class JitSlowLMNLayer(jit.ScriptModule):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh, out_hidden=False):
        super().__init__()
        self.layer = LinearMemoryNetwork(in_size, hidden_size, memory_size, act=act, out_hidden=out_hidden)

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return self.layer.init_hidden(batch_size)

    @jit.script_method
    def forward(self, x, m_prev):
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev, m_prev = self.layer(xt, m_prev)
            out.append(h_prev)
        return torch.stack(out), m_prev

    @jit.script_method
    def hidden_reconstruction(self, x):
        assert len(x.shape) == 3
        m_prev = self.init_hidden(x.shape[1])
        hl = []
        x = x.unbind(0)
        # encode
        for t in range(len(x)):
            xt = x[t]
            h_prev, m_prev = self.layer.encode(xt, m_prev)
            hl.append(h_prev.detach())
        # decode
        reco = []
        for t in range(len(x)):
            h_prev, m_prev = self.decoder(m_prev)
            reco = [h_prev] + reco
        return torch.stack(hl, dim=0), torch.stack(reco, dim=0)


class SlowDiscreteRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, rnn):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.ro = nn.Linear(hidden_size, output_size)
        self.rnn = rnn

    def forward(self, x):
        emb = self.embed(x)
        h0 = self.rnn.init_hidden(x.shape[1])
        ed_out, _ = self.rnn(emb, h0)
        y_out = self.ro(ed_out)

        self._ed_out = ed_out
        return y_out  # logits

    def params_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "output_size": self.output_size
        }


class SlowLSTM(nn.LSTM):
    def __init__(self, **kwargs):
        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs['num_layers']
        super().__init__(**kwargs)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return cuda_move(h0), cuda_move(c0)


class SlowLinearMemoryNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh):
        super().__init__()
        self.memory_size = memory_size
        self.act = act
        self.Wxh = nn.Linear(in_size, hidden_size)
        self.Whm = nn.Linear(hidden_size, memory_size)
        self.Wmh = nn.Linear(memory_size, hidden_size)
        self.Wmm = nn.Linear(memory_size, memory_size)

    def init_hidden(self, batch_size):
        return cuda_move(torch.zeros(batch_size, self.memory_size))

    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(self.Wxh(x_prev) + self.Wmh(m_prev))
        m_curr = self.Whm(h_curr) + self.Wmm(m_prev)
        return m_curr, h_curr


class SlowLMNLayer(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh):
        super().__init__()
        self.layer = SlowLinearMemoryNetwork(in_size, hidden_size, memory_size, act=act)

    def init_hidden(self, batch_size):
        return self.layer.init_hidden(batch_size)

    def forward(self, x, m_prev):
        assert len(x.shape) == 3
        out = []
        for t in range(x.shape[0]):
            xt = x[t]
            m_prev, ht = self.layer(xt, m_prev)
            out.append(m_prev)
        return torch.stack(out), m_prev


def bench():
    set_gpu()
    set_allow_cuda(True)
    T, B, F = 100, 64, 300
    n_trials = 10
    fake_input = cuda_move(torch.zeros(T, B).long())

    print("JitRNN ", end='')
    rnn = RNNLayer(100, 100)
    model = cuda_move(DiscreteRNN(F, F, 100, rnn=rnn))
    model(fake_input)
    foo = lambda: model(fake_input)
    timeit(foo, n_trials)

    print("JitLMN ", end='')
    rnn = LMNLayer(100, 100, 100)
    model = cuda_move(DiscreteRNN(F, F, 100, rnn=rnn))
    model(fake_input)
    foo = lambda: model(fake_input)
    timeit(foo, n_trials)

    print("JitSlowLMN ", end='')
    rnn = LMNLayer(100, 100, 100)
    model = cuda_move(DiscreteRNN(F, F, 100, rnn=rnn))
    model(fake_input)
    foo = lambda: model(fake_input)
    timeit(foo, n_trials)

    print("SlowLMN ", end='')
    rnn = SlowLMNLayer(100, 100, 100)
    model = cuda_move(SlowDiscreteRNN(F, F, 100, rnn=rnn))
    model(fake_input)
    foo = lambda: model(fake_input)
    timeit(foo, n_trials)

    print("SlowLSTM ", end='')
    rnn = SlowLSTM(input_size=100, hidden_size=100, num_layers=1)
    model = cuda_move(SlowDiscreteRNN(F, F, 100, rnn=rnn))
    model(fake_input)
    foo = lambda: model(fake_input)
    timeit(foo, n_trials)


if __name__ == '__main__':
    bench()
