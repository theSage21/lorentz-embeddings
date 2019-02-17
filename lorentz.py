import os
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from tqdm import trange, tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader


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


def set_dim0(x):
    dim0 = torch.sqrt(1 + torch.norm(x[:, 1:], dim=1))
    x[:, 0] = dim0
    return x


# ========================= models


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
                update = set_dim0(update)
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
            set_dim0(self.table.weight)
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
        return loss

    def lorentz_to_poincare(self):
        table = self.table.weight.data.numpy()
        return table[:, 1:] / (
            table[:, :1] + 1
        )  # diffeomorphism transform to poincare ball


class Graph(Dataset):
    def __init__(self, pairwise_matrix, sample_size=10):
        self.pairwise_matrix = pairwise_matrix
        self.n_items = len(pairwise_matrix)
        self.sample_size = sample_size

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        I = torch.Tensor([i]).squeeze().long()
        while True:
            j = random.randint(0, self.n_items - 1)
            if j != i:
                break
        min = self.pairwise_matrix[i, j]
        indices = [
            index
            for index, is_less in enumerate(self.pairwise_matrix[i] < min)
            if is_less
        ][: self.sample_size]
        # offset indices by 1 and pad with 0
        Ks = ([i + 1 for i in [j] + indices] + [0] * self.sample_size)[
            : self.sample_size
        ]
        return I, torch.Tensor(Ks).long()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="File:pairwise_matrix")
    parser.add_argument(
        "-sample_size", help="How many samples in the N matrix", default=10
    )
    parser.add_argument("-batch_size", help="How many samples in the batch", default=32)
    parser.add_argument(
        "-shuffle", help="Shuffle within batch while learning?", default=True
    )
    parser.add_argument(
        "-epochs", help="How many epochs to optimize for?", default=1_000_000
    )
    parser.add_argument(
        "-poincare_dim", help="Poincare projection time. Lorentz will be + 1", default=2
    )
    parser.add_argument(
        "-n_items", help="How many items to embed?", default=None, type=int
    )
    parser.add_argument("-learning_rate", help="RSGD learning rate", default=0.1)
    parser.add_argument("-log_step", help="Log at what multiple of epochs?", default=1)
    parser.add_argument("-logdir", help="What folder to put logs in", default="runs")
    args = parser.parse_args()
    # ----------------------------------- get the correct matrix
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    fl, obj = args.dataset.split(":")

    exec(f"from {fl} import {obj} as pairwise")
    args.n_items = len(pairwise) if args.n_items is None else args.n_items
    pairwise = pairwise[: args.n_items, : args.n_items]
    # ---------------------------------- Generate the proper objects

    dataloader = DataLoader(
        Graph(pairwise, args.sample_size),
        shuffle=args.shuffle,
        batch_size=args.batch_size,
    )
    net = Lorentz(
        args.n_items, args.poincare_dim + 1
    )  # as the paper follows R^(n+1) for this space
    rsgd = RSGD(net.parameters(), learning_rate=args.learning_rate)
    writer = SummaryWriter(f"{args.logdir}/{args.dataset}  {datetime.utcnow()}")

    for epoch in trange(args.epochs, desc="Epochs", ncols=80):
        with tqdm(ncols=80) as pbar:
            for I, Ks in dataloader:
                rsgd.zero_grad()
                loss = net(I, Ks).mean()
                loss.backward()
                rsgd.step()
                pbar.set_description(f"Batch Loss: {float(loss)}")
                if torch.isnan(loss) or torch.isinf(loss):
                    break
            writer.add_scalar("loss", loss, epoch)
