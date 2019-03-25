import time

import torch
import numpy as np
from torch.autograd import Variable
import functools
import inspect
import warnings

ALLOW_CUDA = True  # Global variable to control cuda_move allocation behavior


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    source: https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
    """
    string_types = (type(b''), type(u''))
    if isinstance(reason, string_types):
        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason
        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."
        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2
    else:
        raise TypeError(repr(type(reason)))


def set_allow_cuda(b):
    global ALLOW_CUDA
    ALLOW_CUDA = b


def set_gpu():
    import os
    try:
        import gpustat
    except ImportError as e:
        print("gpustat module is not installed.")
        raise e

    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]

    print("Setting GPU to: {}".format(bestGPU))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)


def cuda_move(args):
    """ Move a sequence of tensors to CUDA if the system supports it. """
    if not ALLOW_CUDA:
        return args
    b = torch.cuda.is_available()
    # for t in args:
    #     if b:
    #         yield t.cuda()
    #     else:
    #         yield t
    if b:
        return args.cuda()
    else:
        return args


def gradient_clipping(model, clip=1):
    """ Clip the value of each gradient component.
        Args:
            clip: maximum absolute value
    """
    for name, p in model.named_parameters():
        try:
            p.grad.data.clamp_(min=-clip, max=clip)
        except AttributeError as e:
            # print("Parameter {} has no gradient.".format(name))
            # raise e
            pass


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


def test_trial(foo, k, epsilon=1.e-6):
    errs = []
    for _ in range(k):
        e = foo()
        errs.append(e)
    errs = np.array(errs)
    mean_err = errs.mean()
    std_err = errs.std()
    print("{} mean_error {}, std_error {}".format(foo.__name__, mean_err, std_err))
    assert mean_err < epsilon


def lock_gpu(gpu_id, ngb=10):
    """
        Allocates some memory and lock a GPU waiting user input. Only useful to reserve GPU memory
        if Tensorflow is not behaving properly.
    """
    GB = ngb * (10 ** 9)
    a = torch.zeros(GB)
    a.cuda(gpu_id)
    print("GPU {} is currenytly locked. Allocated {} GB.".format(gpu_id, ngb))
    input()


def timeit(foo, n_trials):
    times = []
    for _ in range(n_trials):
        start = time.time()
        foo()
        end = time.time()
        t = end - start
        times.append(t)
    t_min = min(times)
    t_mean = sum(times) / n_trials
    print("min: {:.5f}, mean: {:.5f}, n_trials: {}".format(t_min, t_mean, n_trials))
