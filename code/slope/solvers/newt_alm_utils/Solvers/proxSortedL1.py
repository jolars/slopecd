import numpy as np

from slope.solvers.newt_alm_utils.mexfun import mexssgn, proxSortedL1Mex


def proxSortedL1(y, lambda_lam):
    """Normalization"""
    m, n = lambda_lam.shape
    my, ny = y.shape
    lambda_lam = lambda_lam.T.reshape(m * n, 1)
    y = y.T.reshape(my * ny, 1)
    sgn = mexssgn.mexssgn(y)
    yy = sorted(abs(y), reverse=True)
    yy = np.array(yy)
    idx = list(np.argsort(-abs(y.reshape(my * ny))))
    y = yy
    """ Simplify the problem """
    y_lam = list(np.where(y > lambda_lam)[0])
    if not y_lam:
        k = y_lam
    else:
        k = y_lam[-1] + 1
    """ Compute solution and re-normalize """
    n = len(y)
    x = np.zeros([n, 1])
    rr2 = np.ones([n, 1])
    Bmap = lambda x: x - np.vstack((x[1:], [0]))
    nv = []
    if k:
        v1 = y[:k]
        v2 = lambda_lam[:k]
        v = proxSortedL1Mex.proxSortedL1Mex(v1, v2)
        nv = v > 1e-20
        x[idx[:k]] = v
        vv = Bmap(v)
        rr2[:k] = abs(vv) < 1e-12
    """ Restore signs """
    x = sgn * x

    class info:
        pass

    info.idx = idx
    info.rr2 = rr2
    info.k = k
    info.s = sgn
    info.nz = sum(nv)
    return x, info
