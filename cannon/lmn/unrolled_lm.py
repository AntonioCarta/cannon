import random
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from ..callbacks import TBCallback
from ..utils import cuda_move
from ..functional import selu


class UnrolledLM(nn.Module):
    def __init__(self, in_size, k, hidden_size, out_size, activation_fun=selu):
        super().__init__()
        self.activation_fun = activation_fun
        self.in_size = in_size
        self.k = k
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.Wxh = nn.Linear(in_size, hidden_size)
        self.Whh = nn.Linear((k - 1) * hidden_size, hidden_size, bias=False)
        self.Wmo = nn.Linear(k * hidden_size, out_size, bias=False)

    def forward(self, X):
        batch_size = X.size(1)
        n_steps = X.size(0)

        outs = []
        h_prevs = deque([cuda_move(torch.zeros(batch_size, self.hidden_size)) for _ in range(self.k-1)])
        for t in range(n_steps):
            X_t = X[t]

            h_cat = torch.cat(list(reversed(h_prevs)), dim=1)
            # h_t = F.tanh(self.Whh(h_cat) + self.Wxh(X_t))
            h_t = self.activation_fun(self.Whh(h_cat) + self.Wxh(X_t))

            h_out = torch.cat([h_t] + list(reversed(h_prevs)), dim=1)
            out_t = self.Wmo(h_out)
            outs.append(out_t)

            h_prevs.append(h_t)
            h_prevs.popleft()  # Move the window
            h_prevs[0] = h_prevs[0].data, requires_grad=False
        return torch.stack(outs, 0)

    def params_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'k': self.k,
        }


class ULMTBCallback(TBCallback):
    def __init__(self, log_dir, input_dim=None):
        super().__init__(log_dir, input_dim)

    def after_epoch(self, model_trainer, train_data, validation_data):
        n_iter = model_trainer.global_step
        if n_iter % model_trainer.validation_steps == 0:
            self._save_activations(model_trainer, train_data)
        super().after_epoch(model_trainer, train_data, validation_data)

    def _save_activations(self, model_trainer, train_data):
        n_iter = model_trainer.global_step
        random_shuffle = list(train_data.get_one_hot_list())
        random.shuffle(random_shuffle)

        n_samples = 100
        activs = []
        for X_i, y_i in random_shuffle[1:n_samples]:
            X_data = cuda_move(X_i)
            act_i = self._collect_sample_activations(model_trainer.model, X_data)
            activs.append(act_i)

        act_stack = [[] for _ in activs[0]]
        for act in activs:
            for i, el in enumerate(act):
                act_stack[i].append(el)
        mean_act = []
        for el in act_stack:
            mean_act.append(torch.cat(el, 0))
        for name, el in zip(('hidden', 'output'), mean_act):
            self.writer.add_histogram('activation/' + name, el.clone().cpu().data.clamp_(-10**6, 10**6).numpy(), n_iter, bins='sturges')

    def _collect_sample_activations(self, lm, X):
        activs = []
        n_steps = X.size(0)
        batch_size = X.size(1)

        h_acts = []
        outs = []
        h_prevs = deque([cuda_move(Variable(torch.zeros(batch_size, lm.hidden_size))) for _ in range(lm.k-1)])
        for t in range(n_steps):
            X_t = X[t]
            h_cat = torch.cat(list(reversed(h_prevs)), dim=1)
            # h_t = F.tanh(lm.Whh(h_cat) + lm.Wxh(X_t))
            h_t = selu(lm.Whh(h_cat) + lm.Wxh(X_t))
            h_acts.append(h_t)

            h_out = torch.cat([h_t] + list(reversed(h_prevs)), dim=1)
            out_t = F.sigmoid(lm.Wmo(h_out))
            outs.append(out_t)

            h_prevs.append(h_t)
            h_prevs.popleft()  # Move the window
            h_prevs[0] = Variable(h_prevs[0].data, requires_grad=False)

        activs.append(torch.stack(h_prevs, 0))
        activs.append(torch.stack(outs, 0))
        return activs

    def __str__(self):
        return "LMTBCallback(logdir={})".format(self.log_dir)
