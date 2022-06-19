"""
Various permuation, ordering objects

"""
from scipy import sparse
import numpy as np



def permutation_matrix(x):
    n = len(x)

    signs = nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = sparse.lil_matrix((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[ord_j]

    return sparse.csc_array(pi)

def nonzero_sign(x):
    n = len(x)
    out = np.empty(n)

    for i in range(n):
        s = np.sign(x[i])
        out[i] = s if s != 0 else 0

    return out

# inverse of the matrix B.T
def BTinv(x):
    return(np.cumsum(x))


# inverse of the matrix B
def Binv(x):
    return( np.cumsum(x[::-1])[::-1])

def B(x):
    y = x.copy()
    y[:-1] -= x[1:]
    return(y)

#Bt^-1B^-1
# returns (BBt) ^-1 B x
def BBT_inv_B(x):
    return( BTinv(x))