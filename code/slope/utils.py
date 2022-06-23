import numpy as np
import scipy.sparse as sparse
from numba import njit
from numpy.linalg import norm
from sklearn.isotonic import isotonic_regression


@njit
def ST(x, u):
    if x > u:
        return x - u
    elif x < -u:
        return x + u
    else:
        return 0


def primal(residual, beta, lambdas):
    n = len(residual)
    return (norm(residual) ** 2) / (2 * n) + np.sum(
        lambdas * np.sort(np.abs(beta))[::-1]
    )


def dual(theta, y):
    n = len(y)
    return (norm(y) ** 2 - norm(y - theta * n) ** 2) / (2 * n)


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


def get_clusters(w):
    # check if there is a cheaper way of doing this
    unique, counts = np.unique(np.abs(w), return_inverse=False, return_counts=True)
    cluster_indices = np.argsort(np.abs(w))[::-1]
    cluster_ptr = np.cumsum(counts[::-1])
    cluster_ptr = np.r_[0, cluster_ptr]
    return cluster_indices, cluster_ptr, unique[::-1]


@njit
def slope_threshold(x, lambdas, cluster_ptr, c, n_c, j):
    cluster_size = cluster_ptr[j + 1] - cluster_ptr[j]

    abs_x = abs(x)
    sign_x = np.sign(x)

    # check which direction we need to search
    up_direction = abs_x - sum(lambdas[cluster_ptr[j] : cluster_ptr[j + 1]]) > np.abs(
        c[j]
    )

    if up_direction:
        start = cluster_ptr[j + 1]
        lo = sum(lambdas[start : start + cluster_size])

        for k in range(j, -1, -1):
            start = cluster_ptr[k]
            hi = sum(lambdas[start : start + cluster_size])

            abs_c_k = abs(c[k])

            if abs_x < lo + abs_c_k:
                # we must be between clusters
                return x - sign_x * lo, k + 1

            elif abs_x <= hi + abs_c_k:
                # we are in a cluster
                return sign_x * abs_c_k, k

            # replace lower interval by higher before next iteration
            lo = hi

        return x - sign_x * lo, 0
    else:
        end = cluster_ptr[j + 1]
        hi = sum(lambdas[end - cluster_size : end])

        for k in range(j + 1, n_c):
            end = cluster_ptr[k + 1]
            lo = sum(lambdas[end - cluster_size : end])

            abs_c_k = abs(c[k])

            if abs_x > hi + abs_c_k:
                # we must be between clusters
                return x - sign_x * hi, k - 1
            elif abs_x >= lo + abs_c_k:
                # we are in a cluster
                return sign_x * abs_c_k, k

            hi = lo

        if abs_x > hi:
            return x - sign_x * hi, n_c - 1
        else:
            # in zero cluster
            return 0.0, n_c - 1


def add_intercept_column(X):
    n = X.shape[0]

    if sparse.issparse(X):
        return sparse.hstack((sparse.csc_array(np.ones((n, 1))), X), format="csc")
    else
        return np.hstack((np.ones((n, 1)), X))
