import torch


def soft_ortho_constraint(W):
    assert len(W.shape) == 2
    wwt = W @ W.transpose(0, 1)
    I = torch.eye(W.shape[0], device=W.device)
    wwt = wwt - I
    return torch.norm(wwt) ** 2


class OrthogonalPenalty:
    def __init__(self, alpha, ortho_mat_list):
        assert alpha >= 0
        self.alpha = alpha
        self.ortho_mat_list = ortho_mat_list

    def __call__(self, model, x, y):
        ortho_p = 0
        for mat in self.ortho_mat_list:
            ortho_p += soft_ortho_constraint(mat)
        return self.alpha * ortho_p

    def __str__(self):
        return f"OrthogonalPenalty({self.alpha})"