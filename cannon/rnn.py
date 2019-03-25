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
from torch.jit import Tensor


class LinearMemoryNetwork(jit.ScriptModule):
    __constants__ = ['memory_size', 'out_hidden']

    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh, out_hidden=False):
        super().__init__()
        self.out_hidden = out_hidden
        self.memory_size = memory_size
        self.act = act
        self.Wxh = nn.Parameter(0.01 * torch.randn(hidden_size, in_size))
        self.Whm = nn.Parameter(0.01 * torch.randn(memory_size, hidden_size))
        self.Wmm = nn.Parameter(0.01 * torch.randn(memory_size, memory_size))
        self.Wmh = nn.Parameter(0.01 * torch.randn(hidden_size, memory_size))
        self.bh = nn.Parameter(0.01 * torch.randn(hidden_size))
        self.bm = nn.Parameter(0.01 * torch.randn(memory_size))

    @jit.script_method
    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.memory_size, device=self.Wxh.device)

    @jit.script_method
    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev, self.Wmh.t())+ self.bh)
        m_curr = torch.mm(h_curr, self.Whm.t()) + torch.mm(m_prev, self.Wmm.t()) + self.bm
        out = h_curr if self.out_hidden else m_curr
        return out, m_curr


class LMNLayer(jit.ScriptModule):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh):
        super().__init__()
        self.layer = LinearMemoryNetwork(in_size, hidden_size, memory_size, act=act)

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
