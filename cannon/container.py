import torch
from torch import jit
from torch import nn
from .utils import standard_init


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
