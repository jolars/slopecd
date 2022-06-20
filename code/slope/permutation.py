"""
Various permuation, ordering objects

"""
import numpy as np
from scipy import sparse


def nonzero_sign(x):
    n = len(x)
    out = np.empty(n)

    for i in range(n):
        s = np.sign(x[i])
        out[i] = s if s != 0 else 0

    return out


def permutation_matrix(x):
    n = len(x)

    signs = nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = sparse.lil_matrix((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[ord_j]

    return sparse.csc_array(pi)


# build the signedpermutation object
def build_pi(x):

    n = len(x)
    pi_list = np.empty((n, 2), dtype=np.int64)
    piT_list = np.empty((n, 2), dtype=np.int64)
    signs = nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    for j, ord_j in enumerate(order):
        pi_list[j, 0] = ord_j
        pi_list[j, 1] = signs[ord_j]
        piT_list[ord_j, 0] = j
        piT_list[ord_j, 1] = signs[ord_j]

    return pi_list, piT_list


# multiplaction of signed permuation
def pix(x, pi_list):
    return x[pi_list[:, 0]] * pi_list[:, 1]


# inverse of the matrix B.T
def BTinv(x):
    return np.cumsum(x)


# inverse of the matrix B
def Binv(x):
    return np.cumsum(x[::-1])[::-1]


def B(x):
    y = x.copy()
    y[:-1] -= x[1:]
    return y


# Bt^-1B^-1
# returns (BBt) ^-1 B x
def BBT_inv_B(x):
    return BTinv(x)
