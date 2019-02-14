import torch
import random
import numpy as np
from torch import nn
from torch import optim
import matplotlib.pyplot as plt


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
    tn_expand = tn.repeat(1, x.size()[-1])
    result = torch.cosh(tn) * x + torch.sinh(tn) * (v / tn)
    result = torch.where(tn_expand > 0, result, x)  # only update if tangent norm is > 0
    return result


class RSGD(optim.Optimizer):
    def __init__(self, params, learning_rate=None):
        learning_rate = learning_rate if learning_rate is not None else 0.01
        defaults = {"learning_rate": learning_rate}
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                B, D = p.size()
                gl = torch.eye(D, device=p.device, dtype=p.dtype)
                gl[0, 0] = -1
                grad_norm = torch.norm(p.grad.data)
                h = (p.grad.data / grad_norm) @ gl
                proj = (
                    h
                    - (
                        lorentz_scalar_product(p, h) / lorentz_scalar_product(p, p)
                    ).unsqueeze(1)
                    * p
                )
                grad_norm = torch.norm(p.grad.data, dim=1).unsqueeze(1).repeat(1, D)
                update = exp_map(p, -group["learning_rate"] * proj)
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                dim0 = torch.sqrt(1 + torch.norm(update[:, 1:], dim=1))
                update[:, 0] = dim0
                p.data.copy_(update)


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
            # self.table.weight[0] = 0  # padding idx

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
        ui = torch.stack([self.table(I)] * n_ks, dim=1)
        uks = self.table(Ks)
        # ---------- reshape for calculation
        B, N, D = ui.size()
        ui = ui.reshape(B * N, D)
        uks = uks.reshape(B * N, D)
        dists = -lorentz_scalar_product(ui, uks)
        dists = torch.where(dists < 1, torch.ones_like(dists), dists)
        # sometimes 2 embedding can come very close in R^D.
        # when calculating the lorenrz inner product,
        # -1 can become -0.99(no idea!), then arcosh will become nan
        dists = -arcosh(dists)
        # ---------- turn back to per-sample shape
        dists = dists.reshape(B, N)
        loss = -(dists[:, 0] - torch.log(torch.exp(dists).sum(dim=1) + 1e-6))
        return loss, self.table.weight.data.numpy()


def lorentz_to_poincare(table):
    return table[:, 1:] / (
        table[:, :1] + 1
    )  # diffeomorphism transform to poincare ball


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
    emb_dim = 2
    net = Lorentz(10, emb_dim + 1)  # as the paper follows R^(n+1) for this space
    r = RSGD(net.parameters(), learning_rate=0.1)

    I = torch.Tensor([1, 2, 3, 4]).long()
    Ks = torch.Tensor([[2, 3, 4, 9], [4, 5, 6, 1], [6, 7, 1, 5], [8, 9, 2, 1]]).long()
    for i in range(4000):
        loss, table = net(I, Ks)
        loss = loss.mean()
        loss.backward()
        print(loss)
        if torch.isnan(loss) or torch.isinf(loss):
            break
        r.step()
    table = lorentz_to_poincare(table)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*table))
    for i, crd in enumerate(table):
        ax.annotate(i, (crd[0], crd[1]))
    plt.scatter(*zip(*table))
    plt.show()
