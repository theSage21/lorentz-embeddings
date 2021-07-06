import numpy as np
import scipy.sparse as sparse
import pickle
from collections import defaultdict


print("Datasets Available")
print("-" * 20)
# ################### Binary Tree
depth = 10 
N = sum(2 ** i for i in range(depth))
mat = np.zeros((N, N))
for i in range(N):
    j = 2 * i + 1
    if j + 1 >= N:
        break
    mat[i, j] = 1
    mat[i, j + 1] = 1
binary_tree = sparse.csr_matrix(mat)
print("Binary Tree      : binary_tree")


# ################### Quad Tree
depth = 4
N = sum(4 ** i for i in range(depth))
mat = np.zeros((N, N))
for i in range(N):
    j = 4 * i + 1
    if j + 3 >= N:
        break
    mat[i, j] = 1
    mat[i, j + 1] = 1
    mat[i, j + 2] = 1
    mat[i, j + 3] = 1
quad_tree = sparse.csr_matrix(mat)
print("Quad Tree        : quad_tree")
print("-" * 20)



def get_pickle_dataset(filename):
    return pickle.load(open(filename, 'rb'))

datasets = {
    'binary_tree': binary_tree,
    'quad_tree':  quad_tree
}

print(' '.join(datasets.keys()))

def get_dataset(dataset_name_or_filename):
    print("using {} dataset".format(dataset_name_or_filename))
    if dataset_name_or_filename in datasets.keys():
        print("using builtin dataset")
        return datasets[dataset_name_or_filename]
    else:
        return get_pickle_dataset(dataset_name_or_filename)

