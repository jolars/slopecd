import warnings

import numpy as np
from numba import njit
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, norm, solve
from scipy.sparse.linalg import cg, spsolve

import slope.permutation as slopep
from slope.solvers.newt_alm_utils.WeightedK.nal import Newt_ALM
from slope.utils import add_intercept_column, prox_slope


def newt_alm2(
    A,
    b,
    lambdas,
    fit_intercept=False,
    preposs=True,
    gap_freq=1,
    tol=1e-6,
    max_epochs=1000,
    max_time=np.inf,
    verbose=False,
):
    m, n = A.shape

    if b.ndim == 1:
        b = np.expand_dims(b, -1)

    if fit_intercept:
        raise ValueError("intercept is not currently supported")
        # A = add_intercept_column(A)
        # n += 1

    if lambdas.ndim == 1:
        lambdas = np.expand_dims(lambdas, -1)

    lambda_lam = lambdas.copy() * m

    if preposs:
        D = np.sqrt(sum(A * A, 1))
        A = A[:, D > 1e-12]
        norg = n
        _, n = A.shape
        # if verbose:
        #     if norg - n > 0:
        #         print("preprocess A: norg = %3.0d, n = %3.0d", norg, n)
        AAt = A @ A.T  # ATA
        R = np.linalg.cholesky(AAt + 1e-15 * np.eye(m))
        idx = [k for (k, val) in enumerate(np.diag(R)) if val < 1e-8]
        ss = set(range(m)).difference(set(idx))
        if len(ss) < m:
            A = A[ss, :]
        morg = m
        m, _ = A.shape
        # if verbose:
        #     if morg - m > 0:
        #         print("morg = %3.0d, m = %3.0d", morg, m)
        b = b[0:m]

    hR = lambda x: A @ x
    hRt = lambda x: A.T @ x
    """ tuning parameters  """
    tau_s = 1e-3

    stoprho = 6
    lambdaorg = lambda_lam
    crho = 1
    lambda_lam = crho * lambdaorg
    jj = 1

    init = np.zeros([n, 1])
    stoptol = 10 ** (-stoprho)
    saveyes = 0
    eigsopt_issym = 1
    Rmap = lambda x: A @ x
    Rtmap = lambda x: A.T @ x
    RRtmap = lambda x: Rmap(Rtmap(x))
    """这里求特征值"""
    AAT = A @ A.T
    Lipeig, Lipvector = np.linalg.eig(AAT)
    Lip = max(Lipeig)

    class nalop:
        pass

    nalop.Lip = Lip
    nalop.stoptol = stoptol
    nalop.runphaseI = 1

    x0 = np.zeros([n, 1])
    xi0 = np.zeros([m, 1])
    u0 = np.zeros([n, 1])

    obj, x, xi, u, _, _, monitor = Newt_ALM.Newt_ALM(
        A,
        b,
        n,
        lambda_lam,
        fit_intercept,
        nalop,
        x0,
        xi0,
        u0,
        gap_freq,
        tol,
        max_epochs,
        max_time,
        verbose,
    )

    primals, gaps, times = monitor.get_results()
    intercept = x[0] if fit_intercept else 0.0

    return x[fit_intercept:], intercept, primals, gaps, times
