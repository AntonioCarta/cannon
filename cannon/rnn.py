import torch
from torch import nn
from typing import Tuple
from torch import Tensor
import random
from .utils import standard_init


class LinearMemoryNetwork(nn.Module):
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
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p, gain=0.9)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.memory_size, device=self.Wxh.device)

    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev, self.Wmh.t()) + self.bh)
        m_curr = torch.mm(h_curr, self.Whm.t()) + torch.mm(m_prev, self.Wmm.t()) + self.bm
        out = h_curr if self.out_hidden else m_curr
        return out, m_curr


class LMNDetachCell(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, p_detach, act=torch.tanh):
        super().__init__()
        self.memory_size = memory_size
        self.p_detach = p_detach
        self.act = act
        self.Wxh = nn.Parameter(torch.randn(hidden_size, in_size), requires_grad=True)
        self.Whm = nn.Parameter(torch.randn(memory_size, hidden_size), requires_grad=True)
        self.Wmm = nn.Parameter(torch.randn(memory_size, memory_size), requires_grad=True)
        self.Wmh = nn.Parameter(torch.randn(hidden_size, memory_size), requires_grad=True)
        self.bh = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.bm = nn.Parameter(torch.randn(memory_size), requires_grad=True)
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p, gain=0.9)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.memory_size, device=self.Wxh.device)

    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        if random.uniform(0, 1) < self.p_detach:
            h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev.detach(), self.Wmh.t()) + self.bh)
        else:
            h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev, self.Wmh.t()) + self.bh)
        m_curr = torch.mm(h_curr, self.Whm.t()) + torch.mm(m_prev, self.Wmm.t()) + self.bm
        return m_curr, m_curr


class RNNCell(nn.Module):
    def __init__(self, in_size, hidden_size, act=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.act = act
        self.Wxh = nn.Parameter(torch.randn(hidden_size, in_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.Wxh.device)

    def forward(self, x_prev, m_prev):
        assert len(x_prev.shape) == 2
        h_curr = self.act(torch.mm(x_prev, self.Wxh.t()) + torch.mm(m_prev, self.Whh.t()) + self.bh)
        return h_curr, h_curr


class LSTMCell(nn.Module):
    def __init__(self, in_size, hidden_size, fbias=0.0):
        """ source: https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py """
        super(LSTMCell, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(0.01 * torch.randn(4 * hidden_size, in_size))
        self.weight_hh = nn.Parameter(0.01 * torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(0.01 * torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(0.01 * torch.randn(4 * hidden_size))

        with torch.no_grad():  # forget bias initialization
            mask = torch.zeros_like(self.bias_hh)
            mask[hidden_size: 2*hidden_size] = 1  # ingate, forgetgate, cellgate, outgate
            self.bias_hh.masked_fill_(mask == 1, fbias)
        assert self.bias_hh.is_leaf

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(batch_size, self.hidden_size, device=self.weight_ih.device),
                torch.zeros(batch_size, self.hidden_size, device=self.weight_ih.device))

    def forward(self, input, state):
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


class RecurrentLayer(nn.Module):
    def __init__(self, layer):
        """
        Generic RNN layer. Takes an entire sequence as input, while cells take a single item.
        Args:
            layer: recurrent cell
        """
        super().__init__()
        self.layer = layer

    def init_hidden(self, batch_size: int) -> Tensor:
        return self.layer.init_hidden(batch_size)

    def forward(self, x, m_prev):
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev, m_prev = self.layer(xt, m_prev)
            out.append(h_prev)
        return torch.stack(out), m_prev


class LMNDetachLayer(RecurrentLayer):
    def __init__(self, in_size, hidden_size, memory_size, p_detach, act=torch.tanh):
        layer = LMNDetachCell(in_size, hidden_size, memory_size, p_detach, act=act)
        super().__init__(layer)


class LMNLayer(RecurrentLayer):
    def __init__(self, in_size, hidden_size, memory_size, act=torch.tanh):
        layer = LinearMemoryNetwork(in_size, hidden_size, memory_size, act=act)
        super().__init__(layer)


class RNNLayer(RecurrentLayer):
    def __init__(self, in_size, hidden_size, act=torch.tanh):
        layer = RNNCell(in_size, hidden_size, act=act)
        super().__init__(layer)


class LSTMLayer(RecurrentLayer):
    def __init__(self, in_size, hidden_size, fbias=0.0):
        layer = LSTMCell(in_size, hidden_size, fbias=fbias)
        super().__init__(layer)


class LSTMDetachLayer(nn.Module):
    def __init__(self, in_size, hidden_size, p_detach):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = nn.LSTMCell(in_size, hidden_size)
        self.p_detach = p_detach

    def init_hidden(self, batch_size: int):
        return (torch.zeros(batch_size, self.hidden_size, device=self.layer.weight_ih.device),
                torch.zeros(batch_size, self.hidden_size, device=self.layer.weight_ih.device))

    def forward(self, x, prev_state):
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            if random.uniform(0, 1) < self.p_detach:
                prev_state = prev_state[0].detach(), prev_state[1]
            prev_state = self.layer(xt, prev_state)
            out.append(prev_state[0])
        return torch.stack(out), prev_state


class SequenceClassifier(nn.Module):
    def __init__(self, rnn_cell, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn_cell
        self.ro = nn.Linear(hidden_size, output_size)
        for p in self.ro.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p)

    def forward(self, x):
        h0 = self.rnn.init_hidden(x.shape[1])
        h_last = self.rnn(x, h0)[0][-1]
        y = self.ro(h_last)
        return y


class ItemClassifier(nn.Module):
    def __init__(self, rnn, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn
        self.ro = nn.Linear(hidden_size, output_size)
        for p in self.ro.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_normal_(p)

    def forward(self, x):
        h0 = self.rnn.init_hidden(x.shape[1])
        h_seq = self.rnn(x, h0)[0]
        self._act_list = h_seq
        y = self.ro(h_seq)
        return y


class DiscreteRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, rnn):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.ro = nn.Linear(hidden_size, output_size)
        self.rnn = rnn
        standard_init(self.parameters())

    def forward(self, x):
        emb = self.embed(x)
        h0 = self.rnn.init_hidden(x.shape[1])
        ed_out, _ = self.rnn(emb, h0)
        y_out = self.ro(ed_out)
        return y_out  # logits