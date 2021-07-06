import os
import sys
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import datasets
import pickle
from lorentz_embeddings.lorentz import *
import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt
plt.style.use("ggplot")


import warnings
warnings.filterwarnings('ignore')


_moon_count = 0


def _moon(loss, phases="🌕🌖🌗🌘🌑🌒🌓🌔"):
    global _moon_count
    _moon_count += 1
    p = phases[_moon_count % 8]
    return f"{p} Loss: {float(loss)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="File:pairwise_matrix")
    parser.add_argument(
        "-sample_size", help="How many samples in the N matrix", default=5, type=int
    )
    parser.add_argument(
        "-batch_size", help="How many samples in the batch", default=32, type=int
    )
    parser.add_argument(
        "-burn_c",
        help="Divide learning rate by this for the burn epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-burn_epochs",
        help="How many epochs to run the burn phase for?",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-plot", help="Plot the embeddings", default=False, action="store_true"
    )
    parser.add_argument("-plot_size", help="Size of the plot", default=3, type=int)
    parser.add_argument(
        "-plot_graph",
        help="Plot the Graph associated with the embeddings",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-overwrite_plots",
        help="Overwrite the plots?",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-ckpt", help="Which checkpoint to use?", default=None, type=str
    )
    parser.add_argument(
        "-shuffle", help="Shuffle within batch while learning?", default=True, type=bool
    )
    parser.add_argument(
        "-epochs", help="How many epochs to optimize for?", default=10, type=int
    )
    parser.add_argument(
        "-poincare_dim",
        help="Poincare projection time. Lorentz will be + 1",
        default=2,
        type=int,
    )
    parser.add_argument(
        "-n_items", help="How many items to embed?", default=None, type=int
    )
    parser.add_argument(
        "-learning_rate", help="RSGD learning rate", default=0.1, type=float
    )
    parser.add_argument(
        "-log_step", help="Log at what multiple of epochs?", default=1, type=int
    )
    parser.add_argument("-log", default=False, type=bool)
    parser.add_argument(
        "-logdir", help="What folder to put logs in", default="runs", type=str
    )
    parser.add_argument(
        "-save_step", help="Save at what multiple of epochs?", default=100, type=int
    )
    parser.add_argument(
        "-savedir", help="What folder to put checkpoints in", default="ckpt", type=str
    )
    parser.add_argument(
        "-loader_workers",
        help="how many workers to generate tensors",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-device",
        default='cuda'
    )
    args = parser.parse_args()
    # ----------------------------------- get the correct matrix
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.dataset.split('.')[:1] == ['pkl'] and os.path.exists(args.dataset):
        pairwise = pickle.load(open(args.dataset, 'rb'))
    else:
        pairwise = datasets.get_dataset(args.dataset)
        pairwise = pairwise[: args.n_items, : args.n_items]
    args.n_items = pairwise.shape[0] if args.n_items is None else args.n_items
    print(f"{args.n_items} being embetasksdded")

    # ---------------------------------- Generate the proper objects
    net = Lorentz(
        args.n_items, args.poincare_dim + 1
    ).to(args.device)  # as the paper follows R^(n+1) for this space
    if args.plot:
        if args.poincare_dim != 2:
            print("Only embeddings with `-poincare_dim` = 2 are supported for now.")
            sys.exit(1)
        if args.ckpt is None:
            print("Please provide `-ckpt` when using `-plot`")
            sys.exit(1)
        if os.path.isdir(args.ckpt):
            paths = [
                os.path.join(args.ckpt, c)
                for c in os.listdir(args.ckpt)
                if c.endswith("ckpt")
            ]
        else:
            paths = [args.ckpt]
        paths = list(sorted(paths))
        edges = [
            tuple(edge)
            for edge in set(
                [
                    frozenset((a + 1, b + 1))
                    for a, row in enumerate(pairwise > 0)
                    for b, is_non_zero in enumerate(row)
                    if is_non_zero
                ]
            )
        ]
        print(len(edges), "nodes")
        internal_nodes = set(
            node
            for node, count in Counter(
                [node for edge in edges for node in edge]
            ).items()
            if count > 1
        )
        edges = np.array([edge for edge in edges if edge[1] in internal_nodes])
        print(len(edges), "internal nodes")
        for path in tqdm(paths, desc="Plotting"):
            save_path = f"{path}.svg"
            if os.path.exists(save_path) and not args.overwrite_plots:
                continue
            net.load_state_dict(torch.load(path))
            table = net.lorentz_to_poincare()
            # skip padding. plot x y
            plt.figure(figsize=(7, 7))
            if args.plot_graph:
                for edge in edges:
                    plt.plot(
                        table[edge, 0],
                        table[edge, 1],
                        color="black",
                        marker="o",
                        alpha=0.5,
                    )
            else:
                plt.scatter(table[1:, 0], table[1:, 1])
            plt.title(path)
            plt.gca().set_xlim(-1, 1)
            plt.gca().set_ylim(-1, 1)
            plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
            plt.savefig(save_path)
            plt.close()
        sys.exit(0)

    graph_dataset = Graph(pairwise, args.sample_size, args.batch_size)
    dataloader = DataLoader(
        graph_dataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
    )
    rsgd = RSGD(net.parameters(), learning_rate=args.learning_rate)

    name = f"{args.dataset}  {datetime.utcnow()}"
    if args.log:
        writer = SummaryWriter(f"{args.logdir}/{name}")

    with tqdm(ncols=80, mininterval=0.2, total=args.epochs) as epoch_bar:
        for epoch in range(args.epochs):
            rsgd.learning_rate = (
                args.learning_rate / args.burn_c
                if epoch < args.burn_epochs
                else args.learning_rate
            )
            for I, Ks in tqdm(dataloader, total=int(np.ceil(graph_dataset.n_items / args.batch_size))):
                I = I.to(args.device)
                Ks = Ks.to(args.device)
                rsgd.zero_grad()
                loss = net(I, Ks).mean()
                loss.backward()
                rsgd.step()
            if args.log:
                writer.add_scalar("loss", loss, epoch)
                writer.add_scalar(
                    "recon_preform", recon(net.get_lorentz_table(), pairwise), epoch
                )
                writer.add_scalar("table_test", net._test_table(), epoch)
            if epoch % args.save_step == 0:
                torch.save(net.state_dict(), f"{args.savedir}/{epoch} {name}.ckpt")
            epoch_bar.set_description(
                f"🔥 Burn phase loss: {float(loss)}"
                if epoch < args.burn_epochs
                else _moon(loss)
            )
            epoch_bar.update(1)
    torch.save(net.state_dict(), args.dataset.replace(".", "_") + "call_graph_hyper_embeddings.pt")
