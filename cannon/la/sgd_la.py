import torch
from torch import nn


class SequentialLinearAutoencoderCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """ Sequential Linear Autoencoder module. """
        super().__init__()
        self.hidden_size = hidden_size
        self.A = nn.Linear(input_size, hidden_size)
        self.B = nn.Linear(hidden_size, hidden_size)
        self.Ch = nn.Linear(hidden_size, hidden_size)
        self.Cx = nn.Linear(hidden_size, input_size)
        self.h0 = nn.Parameter(torch.rand(1, hidden_size))

    def init_hidden(self, batch_size):
        """ Initialize the hidden state.
        Returns: initial hidden state tensor (batch, features)
        """
        return self.h0.expand(batch_size, -1)

    def forward(self, xt, h_prev):
        """ Forward step.
        Args:
            - xt: tensor (batch, features)
            - h_prev: tensor (batch, features)
        """
        return self.A(xt) + self.B(h_prev)

    def decode(self, ht):
        """ Decodes a single timestep.
        Args:
            - ht: tensor (batch, features)
        Returns:
            - h_prev: previous hidden state tensor (batch, features)
            - x_prev: previous input tensor (batch, features)
        """
        return self.Ch(ht), self.Cx(ht)

    def params_dict(self):
        return {'hidden_size': self.hidden_size}
