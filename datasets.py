import numpy as np


print("Datasets Available")
print("-" * 20)
# ################### Binary Tree
depth = 8
N = sum(2 ** i for i in range(depth))
mat = np.zeros((N, N))
for i in range(N):
    j = 2 * i + 1
    if j + 1 >= N:
        break
    mat[i, j] = 1
    mat[i, j + 1] = 1
bin_mat = mat
print("Binary Tree: bin_mat")


# ################### Quad Tree
depth = 5
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
quad_mat = mat
print("Quad Tree: quad_mat")
print("-" * 20)
