"""
Created on Tue Jun 21 10:12:57 2022

@author: Prince_Li
"""

import numpy as np
from numba import njit


@njit
def proxSortedL1Mex(y, lambda_1):
    n = len(y)
    idx_i = np.zeros(n, dtype=np.int64)
    idx_j = np.zeros(n, dtype=np.int64)
    s = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
    k = 0
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = y[i, 0] - lambda_1[i, 0]
        w[k] = s[k]
        while k > 0 and w[k - 1] <= w[k]:
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)
        k = k + 1
    for j in range(k):
        d = w[j]
        if d < 0:
            d = 0
        for i in range(int(idx_i[j]), int(idx_j[j]) + 1):
            x[i] = d

    x_out = np.empty((n, 1))
    x_out[:, 0] = x

    return x_out
