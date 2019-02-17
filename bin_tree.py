import numpy as np


N = 101
mat = np.zeros((N, N))
print(mat.shape)
for i in range(N):
    j = 2 * i + 1
    if j + 1 >= N:
        break
    mat[i, j] = 1
    mat[i, j + 1] = 1
