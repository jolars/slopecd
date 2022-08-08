"""
Various permuation, ordering objects

"""
import numpy as np
from numba import njit
from scipy import sparse


@njit
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
@njit
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
@njit
def pix(x, pi_list):
    return x[pi_list[:, 0]] * pi_list[:, 1]


# inverse of the matrix B.T
def BTinv(x):
    return np.cumsum(x)


# inverse of the matrix B
@njit
def Binv(x):
    return np.cumsum(x[::-1])[::-1]


@njit
def B(x):
    y = x.copy()
    y[:-1] -= x[1:]
    return y


# Bt^-1B^-1
# returns (BBt) ^-1 B x
def BBT_inv_B(x):
    return BTinv(x)


@njit
def assemble_sparse_W(
    nC, GammaC, pi_list, A_data, A_indices, A_indptr, m, fit_intercept
):
    W_row = []
    W_col = []
    W_data = []

    start = 0

    if fit_intercept:
        for i in range(m):
            W_col.append(0)
            W_row.append(i)
            W_data.append(1.0)

    for i in range(nC):
        nCi = GammaC[i] + 1 - start
        for j in range(start, GammaC[i] + 1):
            k = pi_list[j, 0]
            pi_list_j1 = pi_list[j, 1]
            for ind in range(
                A_indptr[k + fit_intercept], A_indptr[k + 1 + fit_intercept]
            ):
                W_row.append(A_indices[ind])
                W_col.append(i + fit_intercept)
                val = pi_list_j1 * A_data[ind]
                if nCi > 1:
                    val /= np.sqrt(nCi)
                W_data.append(val)
        start = GammaC[i] + 1

    return np.array(W_row), np.array(W_col), np.array(W_data)


@njit
def assemble_dense_W(nC, GammaC, pi_list, A, fit_intercept):
    m = A.shape[0]

    W = np.zeros((m, nC + fit_intercept))

    if fit_intercept:
        W[:, 0] = np.ones(m, dtype=np.float64)

    start = 0
    for i in range(nC):
        nCi = GammaC[i] + 1 - start
        for j in range(start, GammaC[i] + 1):
            W[:, i + fit_intercept] += (
                pi_list[j, 1] * A[:, pi_list[j, 0] + fit_intercept]
            )
        if nCi > 1:
            W[:, i + fit_intercept] /= np.sqrt(nCi)
        start = GammaC[i] + 1

    return W
