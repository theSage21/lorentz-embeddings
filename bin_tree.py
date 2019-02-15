import os
import io
import torch
import random
import numpy as np
from torch import nn
from torch import optim
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
        self.table = nn.Embedding(n_items, dim)
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


def insert(n):
    """
    n    : Number of non-leaf node in the binary tree

    will output a set of pairs of nodes, which will represent the tree,
    n = 3 will create a tree 1<-0->2 and will become {(1,0), (2,0)}. (child, parent).
    we assume that the roon node is 0 and after the inserted sequentially.
    It can only create complete binary tree correctly so n should be odd number.
    for n = 5,
        0
       / \
      1   2
     / \
    3   4
    output will be {(2, 0), (1, 0), (3, 1), (4, 1)}
    """
    pairs = list()
    for i in range(n):
        pairs.append((2 * i + 1, i))
        pairs.append((2 * i + 2, i))
    return pairs


def dikhaao(table, loss, epoch):
    table = lorentz_to_poincare(table)
    layers = []
    n_nodes = len(table)
    plt.figure(figsize=(10, 7))
    while sum([1 for layer in layers for node in layer]) < n_nodes:
        limit = 2**len(layers)
        layers.append(table[:limit])
        table = table[limit:]
        plt.scatter(*zip(*layers[-1]), label=f'Layer {len(layers) - 1}')
    plt.title(f'{epoch}: N Nodes {n_nodes} Loss {float(loss)}')
    plt.legend()
    images = list(os.listdir('images'))
    plt.savefig(f'images/{len(images)}.svg')
    plt.close()


if __name__ == "__main__":
    emb_dim = 2
    num_nodes = 1001  # should be odd number
    net = Lorentz(num_nodes, emb_dim + 1)  # as the paper follows R^(n+1) for this space
    r = RSGD(net.parameters(), learning_rate=0.1)
    pairs = insert(num_nodes - (num_nodes + 1) // 2)
    np.random.shuffle(pairs)
    pairs = set(pairs)
    print(pairs)
    I = []
    Ks = []
    arange = np.arange(0, num_nodes)
    for x, y in pairs:
        # we have to parent prediction for binary tree, because if
        # we have a tree like 1<-0->2 if we do child prediction a conflict arises.
        # for 0 we have to predict 1, AND 2!! So for I = 0, Ks will have to be
        # [1, 2] and [2, 1], this creates a conflict. Doing parent prediction is easier
        # because for I = 1 Ks can be [0, 2]! No conflicts
        I.append(x)
        temp_Ks = [y]  # keep the parent in the begining
        temp = np.random.permutation(arange)
        for _ in temp:
            if (x, _) not in pairs and _ != x:
                # make sure there is not edge between _ -> x
                temp_Ks.append(_)
            if (
                len(temp_Ks) == 5
            ):  # sample size of 5, the minimum value of this will depend on num_nodes
                break
        Ks.append(temp_Ks)
    I = torch.tensor(I)
    Ks = torch.tensor(Ks)
    batch_size = 1000
    epochs = 40_00_00_000
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            loss = 0
            j = 0
            while j < len(I):
                loss_batch, table = net(I[j : j + batch_size], Ks[j : j + batch_size])
                j += batch_size
                loss_batch = loss_batch.mean()
                loss_batch.backward()
                loss += loss_batch
                r.step()
            if epoch % 10 == 0:
                dikhaao(table, loss, epoch)
            pbar.set_description(f'{epoch}  :   {float(loss)}')
            pbar.update(1)
            if torch.isnan(loss) or torch.isinf(loss):
                print('NAN / Inf')
                break
