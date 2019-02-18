import numpy as np


depth = 3
N = sum(4 ** i for i in range(depth))
mat = np.zeros((N, N))
print(mat.shape)
for i in range(N):
    j = 4 * i + 1
    if j + 3 >= N:
        break
    mat[i, j] = 1
    mat[i, j + 1] = 1
    mat[i, j + 2] = 1
    mat[i, j + 3] = 1
print(mat)
