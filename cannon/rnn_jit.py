"""
    RNN models implemented using torch jit.

    Improved performance of nn.Module alternatives by using Pytorch JIT.
    NOTE: serialization may not work for some models.

    Currently Jit models are about 3x faster.
    CuDNN LSTM is still 3x faster than a comparable LMN.

    JitLMN min: 0.00875, mean: 0.00896, n_trials: 10
    SlowLMN min: 0.02461, mean: 0.03130, n_trials: 10
    SlowLSTM min: 0.00317, mean: 0.00329, n_trials: 10
"""
import torch
from torch import jit
from torch import nn
from torch import Tensor
from typing import Tuple
from .utils import standard_init


class LinearMemoryNetwork(jit.ScriptModule):
    __constants__ = ['memory_size', 'out_hidden']

    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh, out_hidden=False):
        super().__init__()
        self.out_hidden = out_hidden
        self.memory_size = memory_size
        self.act = act
        self.Wxh = nn.Parameter(torch.randn(hidden_size, in_size))
        self.Whm = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.Wmm = nn.Parameter(torch.randn(memory_size, memory_size))
        self.Wmh = nn.Parameter(torch.randn(hidden_size, memory_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))
        self.bm = nn.Parameter(torch.randn(memory_size))
        standard_init(self.parameters())

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.memory_size, device=self.Wxh.device)

    @jit.script_method
    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev, self.Wmh.t()) + self.bh)
        m_curr = torch.mm(h_curr, self.Whm.t()) + torch.mm(m_prev, self.Wmm.t()) + self.bm
        out = h_curr if self.out_hidden else m_curr
        return out, m_curr


class LMNLayer(jit.ScriptModule):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh, out_hidden=False):
        super().__init__()
        self.layer = LinearMemoryNetwork(in_size, hidden_size, memory_size, act=act, out_hidden=out_hidden)
        standard_init(self.parameters())

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


class RNNCell(jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, in_size, hidden_size, act=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.act = act
        self.Wxh = nn.Parameter(0.01 * torch.randn(hidden_size, in_size))
        self.Whh = nn.Parameter(0.01 * torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(0.01 * torch.randn(hidden_size))

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.Wxh.device)

    @jit.script_method
    def forward(self, x_prev, h_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(h_prev, self.Whh.t()) + self.bh)
        return h_curr


class RNNLayer(jit.ScriptModule):
    def __init__(self, in_size, hidden_size, act=torch.tanh):
        super().__init__()
        self.layer = RNNCell(in_size, hidden_size, act=act)

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return self.layer.init_hidden(batch_size)

    @jit.script_method
    def forward(self, x, h_prev):
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev = self.layer(xt, h_prev)
            out.append(h_prev)
        return torch.stack(out), h_prev


class LSTMCell(jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, in_size, hidden_size):
        """ source: https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py """
        super(LSTMCell, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(0.01 * torch.randn(4 * hidden_size, in_size))
        self.weight_hh = nn.Parameter(0.01 * torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(0.01 * torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(0.01 * torch.randn(4 * hidden_size))

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(batch_size, self.hidden_size, device=self.weight_ih.device),
                torch.zeros(batch_size, self.hidden_size, device=self.weight_ih.device))

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layer = LSTMCell(in_size, hidden_size)

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return self.layer.init_hidden(batch_size)

    @jit.script_method
    def forward(self, x: Tensor, prev_state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev, prev_state = self.layer(xt, prev_state)
            out.append(h_prev)
        return torch.stack(out), prev_state


class DiscreteRNN(jit.ScriptModule):
    def __init__(self, input_size, output_size, hidden_size, rnn):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.ro = nn.Linear(hidden_size, output_size)
        self.rnn = rnn
        standard_init(self.parameters())

    @jit.script_method
    def forward(self, x):
        emb = self.embed(x)
        h0 = self.rnn.init_hidden(x.shape[1])
        ed_out, _ = self.rnn(emb, h0)
        y_out = self.ro(ed_out)
        return y_out  # logits


class SequenceClassifier(jit.ScriptModule):
    def __init__(self, rnn_cell, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn_cell
        self.Wo = nn.Parameter(0.01 * torch.rand(output_size, hidden_size))
        self.bo = nn.Parameter(0.01 * torch.rand(output_size))
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_uniform_(p)

    @jit.script_method
    def forward(self, x):
        h0 = self.rnn.init_hidden(x.shape[1])
        h_last = self.rnn(x, h0)[0][-1]
        y = torch.mm(h_last, self.Wo.t()) + self.bo
        return y
