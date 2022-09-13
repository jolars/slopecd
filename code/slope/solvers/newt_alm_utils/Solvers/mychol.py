import numpy as np
import scipy
from scipy.sparse import csc_matrix


def mychol(M, m):
    class L:
        pass

    pertdiag = 1e-15 * np.ones([m, m])
    M = M + scipy.sparse.csc_matrix(pertdiag)  # scipy.sparse.spdiags(pertdiag,0,m,m)
    del pertdiag
    if scipy.sparse.issparse(M):
        M = M.todense()
    L.matfct_options = "cholesky"
    L.perm = list(range(m))
    L.R = np.linalg.cholesky(M)
    L.R = L.R.T
    return L
