import numpy as np
from numba import njit

from sklearn.isotonic import isotonic_regression


@njit
def ST(x, u):
    if x > u:
        return x - u
    elif x < -u:
        return x + u
    else:
        return 0


def dual_norm_slope(X, theta, alphas):
    """Dual slope norm of X.T @ theta"""
    Xtheta = np.sort(np.abs(X.T @ theta))[::-1]
    taus = 1 / np.cumsum(alphas)
    return np.max(np.cumsum(Xtheta) * taus)


def prox_slope(w, alphas):
    w_abs = np.abs(w)
    idx = np.argsort(w_abs)[::-1]
    w_abs = w_abs[idx]
    # projection onto Km+
    w_abs = isotonic_regression(w_abs - alphas, y_min=0, increasing=False)

    # undo the sorting
    inv_idx = np.zeros_like(idx)
    inv_idx[idx] = np.arange(len(w))

    return np.sign(w) * w_abs[inv_idx]
