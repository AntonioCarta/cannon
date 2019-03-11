from cannon.tasks import CopyTask, AdditionTask
from cannon.utils import set_gpu, set_allow_cuda
import torch
set_gpu()


def test_copy():
    set_allow_cuda(False)
    S, T, K = 10, 5, 2
    batch_size = 3
    data = CopyTask(S, T, K)
    x, y = data.get_batch(batch_size)

    # marker at correct position
    assert (x[T + S - 1] == K + 1).all()

    mask = torch.cat([y[:T + S], x[:S]], dim=0)
    out = torch.zeros((T + 2*S, batch_size, K + 2))

    # optimal solution has zero error (logits should be +inf).
    for t in range(2*S + T):
        for b in range(batch_size):
            out[t, b, mask[t, b]] = 10**9
    loss = data.score(out, y)
    assert loss == 0.0


def test_addition():
    seq_len = 10
    batch_size = 4

    data = AdditionTask(seq_len)
    x, y = data.get_batch(batch_size)

    # two and only two positions are marked for each sequence
    for b in range(batch_size):
        assert x[:, b, 1].sum() == 2.0

    # optimal solution has zero error
    out = (x[:, :, 0] * x[:, :, 1]).sum(dim=0).reshape(batch_size, 1)
    assert data.score(out, y) == 0.0


if __name__ == '__main__':
    print("main.")
    test_copy()
    test_addition()

