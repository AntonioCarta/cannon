import torch
import numpy as np
from torch.autograd import Variable


def cuda_move(t):
    """ Move tensor t to CUDA if the system supports it. """
    if torch.cuda.is_available():
        return t.cuda()
    else:
        return t

def gradient_clipping(parameters, clip=10):
    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)

def cosine_similarity(x1: Variable, x2: Variable, dim: int=1, eps=1e-8) -> Variable:
    """ Compute cosine similarity along given dim. """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def is_nan(v: Variable):
    return np.isnan(np.sum(v.data.cpu().numpy()))


def assert_equals(a, b, eps=1e-6):
    assert torch.sum(torch.ge(torch.abs(a - b), eps)) == 0


def assert_relative_equals(a, b, perc=0.01):
    diff = torch.sum(torch.abs(a - b))
    tot = torch.sum(a + b)
    assert diff / tot < perc
