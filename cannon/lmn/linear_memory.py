import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils import cuda_move
from ..callbacks import TBCallback
from ..nn_utils import selu


class SeLULinearMemory(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.out_size = out_size
        self.Wxh = nn.Linear(in_size, hidden_size)
        self.Whm = nn.Linear(hidden_size, memory_size)
        self.Wmh = nn.Linear(memory_size, hidden_size)
        self.Wmm = nn.Linear(memory_size, memory_size)
        self.Wmo = nn.Linear(memory_size, out_size)

    def forward(self, X, logits=False, m_prev=None):
        n_steps = X.size(0)
        batch_size = X.size(1)

        out = []
        if m_prev is None:
            m_prev = cuda_move(Variable(torch.zeros((batch_size, self.memory_size))))
        for t in range(n_steps):
            X_t = X[t]
            h_t = selu(self.Wxh(X_t) + self.Wmh(m_prev))
            m_t = self.Whm(h_t) + self.Wmm(m_prev)

            if logits:
                o_t = self.Wmo(m_t)
            else:
                o_t = F.sigmoid(self.Wmo(m_t))

            m_prev = m_t
            out.append(o_t)

        self._m_prev = m_prev
        return torch.stack(out)

    def params_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'memory_size': self.memory_size,
        }


class LinearMemory(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.out_size = out_size
        self.Wxh = nn.Linear(in_size, hidden_size)
        self.Whm = nn.Linear(hidden_size, memory_size)
        self.Wmh = nn.Linear(memory_size, hidden_size)
        self.Wmm = nn.Linear(memory_size, memory_size)
        self.Wmo = nn.Linear(memory_size, out_size)

    def forward(self, X, logits=False, m_prev=None):
        n_steps = X.size(0)
        batch_size = X.size(1)

        out = []
        if m_prev is None:
            m_prev = cuda_move(Variable(torch.zeros((batch_size, self.memory_size))))
        for t in range(n_steps):
            X_t = X[t]
            h_t = F.tanh(self.Wxh(X_t) + self.Wmh(m_prev))
            m_t = self.Whm(h_t) + self.Wmm(m_prev)

            if logits:
                o_t = self.Wmo(m_t)
            else:
                o_t = F.sigmoid(self.Wmo(m_t))

            m_prev = m_t
            out.append(o_t)

        self._m_prev = m_prev
        return torch.stack(out)

    def params_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'memory_size': self.memory_size,
        }


class LinearMemoryAlt(nn.Module):
    def __init__(self, in_size, hidden_size, memory_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.out_size = out_size
        self.Wxh = nn.Linear(in_size, hidden_size)
        self.Whm = nn.Linear(hidden_size, memory_size)
        self.Wmh = nn.Linear(memory_size, hidden_size)
        self.Wmm = nn.Linear(memory_size, memory_size)
        self.Who = nn.Linear(hidden_size, out_size)

    def forward(self, X, logits=False, m_prev=None):
        n_steps = X.size(0)
        batch_size = X.size(1)

        out = []
        if m_prev is None:
            m_prev = cuda_move(Variable(torch.zeros((batch_size, self.memory_size))))
        for t in range(n_steps):
            X_t = X[t]
            h_t = F.tanh(self.Wxh(X_t) + self.Wmh(m_prev))
            m_t = self.Whm(h_t) + self.Wmm(m_prev)

            if logits:
                o_t = self.Who(h_t)
            else:
                o_t = F.sigmoid(self.Who(h_t))

            m_prev = m_t
            out.append(o_t)

        self._m_prev = m_prev
        return torch.stack(out)

    def params_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'memory_size': self.memory_size,
        }


class LMTBCallback(TBCallback):
    def __init__(self, log_dir, input_dim=None):
        super().__init__(log_dir, input_dim)

    def after_epoch(self, model_trainer, train_data, validation_data):
        n_iter = model_trainer.global_step
        if n_iter % model_trainer.validation_steps == 0:
            tx = self._approx_entropy(model_trainer, train_data, validation_data)
            self.writer.add_scalar('data/entropy', tx.clone().cpu().data.numpy(), n_iter)
            self._save_activations(model_trainer, train_data)
        super().after_epoch(model_trainer, train_data, validation_data)

    def _approx_entropy(self, model_trainer, train_data, validation_data):
        random_shuffle = list(train_data.get_one_hot_list())
        random.shuffle(random_shuffle)
        n_samples = 100
        activs = []
        tx = 0
        for X_i, _ in random_shuffle[1:n_samples]:
            X_data = cuda_move(X_i)
            y_out = model_trainer.model(X_data)
            tx += F.binary_cross_entropy_with_logits(y_out, y_out)
        tx = tx / n_samples
        return tx

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

        for name, el in zip(('hidden', 'memory', 'output'), mean_act):
            t_el = el.clone().cpu().data.clamp_(-10**6, 10**6).numpy()
            t_el[t_el != t_el] = 10 ** 7  # mask NaN
            self.writer.add_histogram('activation/' + name, t_el, n_iter, bins='sturges')

    def _collect_sample_activations(self, lm, X):
        activs = []
        n_steps = X.size(0)
        batch_size = X.size(1)

        out = []
        m_prev = cuda_move(Variable(torch.zeros((batch_size, lm.memory_size))))
        h_prevs, m_prevs, o_prevs = [], [], []
        for t in range(n_steps):
            X_t = X[t]
            h_t = selu(lm.Wxh(X_t) + lm.Wmh(m_prev))
            m_t = lm.Whm(h_t) + lm.Wmm(m_prev)
            o_t = F.sigmoid(lm.Wmo(m_t))

            m_prev = m_t
            h_prevs.append(h_t)
            m_prevs.append(m_t)
            out.append(o_t)

        activs.append(torch.stack(h_prevs))
        activs.append(torch.stack(m_prevs))
        activs.append(torch.stack(out))
        return activs

    def __str__(self):
        return "LMTBCallback(logdir={})".format(self.log_dir)
