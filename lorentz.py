import torch
import random
import numpy as np
from torch import nn
from torch import optim


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def lorentz_scalar_product(x, y):
    # BD, BD -> B
    m = x * y
    result = m[:, 1:].sum(dim=1) - m[:, 0]
    return result


def tangent_norm(x):
    # BD -> B
    return torch.sqrt(lorentz_scalar_product(x, x))


def exp_map(x, v):
    # BD, BD -> BD
    tn = tangent_norm(v).unsqueeze(dim=1)
    result = torch.cosh(tn) * x + torch.sinh(tn) * (v / tn)
    return result


class RSGD(optim.Optimizer):
    def __init__(self, params, learning_rate=None):
        learning_rate = learning_rate if learning_rate is not None else 0.1
        defaults = {"learning_rate": learning_rate}
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                B, D = p.size()
                gl = torch.ones((D, D), device=p.device, dtype=p.dtype)
                gl[0, 0] = -1
                h = p.grad.data @ gl  # NOTE: I don't know how this might work out
                proj = h + lorentz_scalar_product(p, h).unsqueeze(dim=1) * p
                p.data.copy_(exp_map(p, -group["learning_rate"] * proj))


class Lorentz(nn.Module):
    """
    This will embed `n_items` in a `dim` dimensional lorentz space.
    """
    def __init__(self, n_items, dim, init_range=0.001):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.uniform_(self.table.weight, -init_range, init_range)
        # equation 6
        with torch.no_grad():
            dim0 = torch.sqrt(1 + torch.norm(self.table.weight[:, 1:], dim=1))
            self.table.weight[:, 0] = dim0
            self.table.weight[0] = 0  # padding idx

    def forward(self, I, Ks):
        """
        Using the pairwise similarity matrix, generate the following inputs and
        provide to this function.

        Inputs:
            - I     :   - long tensor
                        - size (B,)
                        - This denotes the `i` used in all equations.
            - Ks    :   - long tensor
                        - size (B, N)
                        - This denotes at max `N` documents which come from the
                          nearest neighbor sample.
                        - The `j` document must be the first of the N indices.
                          This is used to calculate the losses
        Return:
            - size (B,)
            - Ranking loss calculated using
              document to the given `i` document.

        """
        n_ks = Ks.size()[1]
        ui = torch.stack([self.table(I)]*n_ks, dim=1)
        uks = self.table(Ks)
        # ---------- reshape for calculation
        B, N, D = ui.size()
        ui = ui.reshape(B*N, D)
        uks = uks.reshape(B*N, D)
        dists = torch.exp(-arcosh(-lorentz_scalar_product(ui, uks)))
        # ---------- turn back to per-sample shape
        dists = dists.reshape(B, N)
        loss = torch.log(dists[:, 0] / dists.sum(dim=1))
        return loss


def N_sample(matrix, i, j, n):
    """
    - Matrix    : is a simple pairwise similarity matrix
    - i         : primary document
    - j         : secondary document
    - n         : Sample n items maximum from the matrix

    0 is a padding index
    """
    min = matrix[i, j]
    indices = [
        index for index, is_less in enumerate(self.sim_matrix[i] < min) if is_less
    ][:n]
    return ([i + 1 for i in [j] + indices] + [0] * n)[:n]


if __name__ == "__main__":
    net = Lorentz(10, 2)
    r = RSGD(net.parameters())

    I = torch.Tensor([1, 2, 2]).long()
    Ks = torch.Tensor([[1, 2, 1, 2],
                       [1, 0, 0, 0],
                       [2, 1, 2, 0]]).long()
    loss = net(I, Ks)
    loss = loss.mean()
    loss.backward()

    r.step()
