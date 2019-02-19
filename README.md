Lorentz Embeddings
==================


A pytorch implementation of [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/pdf/1806.03417.pdf?noredirect=1).

Usage
=====

Binary tree embedding and visualization.

```bash
python lorentz.py --help
python lorentz.py bin_mat  # run binary tree
# plot the checkpoint's embeddings for all saved checkpoints
# in poincare space
python lorentz.py bin_mat -plot -ckpt ckpt
```

To embed an arbitrary graph

1. Add a numpy matrix in the `datasets.py` file with a unique name (`my_graph` for example). This represents a directed adjacency matrix
2. Now you can simply call `python lorentz.py my_graph` to embed your graph.
3. You can use tensorboard to watch the progress with `tensorboard --logdir runs`.
4. You can plot the embeddings using `python lorentz.py my_graph -plot -ckpt ckpt`


For anything else `python lorentz.py --help`
