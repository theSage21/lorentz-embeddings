Lorentz Embeddings
==================


A pytorch implementation of [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/pdf/1806.03417.pdf?noredirect=1).

Usage
=====

Binary tree embedding and visualization.

```bash
mkdir logs
python lorentz.py bin_tree:mat  # run binary tree
```

To embed an arbitrary graph, create a file (say `mygraph.py`) and write code in
it to generate a pairwise similarity matrix. Take a look at the existing code
if you need help. For example, the `bin_tree.py` file generates a numpy matrix
called `mat`. Let's say your file also calls the matrix `mat`.

Now you can simply call `python lorentz.py mygraph:mat` to embed your graph.

You can use tensorboard to watch the progress.
