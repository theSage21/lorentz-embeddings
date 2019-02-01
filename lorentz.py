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


class Net(nn.Module):
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
        """Given Is and the sampled Ks, the system returns probabilities of
        nearest neighbor"""
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
        probas = torch.log(dists / (dists.sum(dim=1).unsqueeze(dim=1)))
        return probas


def N_sample(i, j):
    min = self.sim_matrix[i, j]
    indices = [
        index for index, is_less in enumerate(self.sim_matrix[i] < min) if is_less
    ] + [j]
    return indices


if __name__ == "__main__":
    net = Net(10, 2)
    r = RSGD(net.parameters())

    I = torch.Tensor([1, 2, 2]).long()
    Ks = torch.Tensor([[1, 2, 1, 2],
                       [1, 0, 0, 0],
                       [2, 1, 2, 0]]).long()
    loss = net(I, Ks) - 1
    loss = loss.sum()
    loss.backward()

    r.step()
