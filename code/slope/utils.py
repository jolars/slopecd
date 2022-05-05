import numpy as np
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
    unique, counts = np.unique(
        np.abs(w), return_inverse=False, return_counts=True
    )
    cluster_indices = np.argsort(np.abs(w))[::-1]
    cluster_ptr = np.cumsum(counts[::-1])
    cluster_ptr = np.r_[0, cluster_ptr]
    return cluster_indices, cluster_ptr, unique[::-1]


@njit
def slope_threshold(x, lambdas, cluster_indices, cluster_ptr, c, n_c, j):
    A = cluster_indices[cluster_ptr[j]:cluster_ptr[j + 1]]
    cluster_size = len(A)

    # zero_cluster_size = 0 if c[-1] != 0 else len(C[-1])
    zero_lambda_sum = np.sum(
        lambdas[::-1][np.arange(cluster_size)])

    if np.abs(x) < zero_lambda_sum:
        return 0.0, n_c - 1

    lo = zero_lambda_sum
    hi = zero_lambda_sum

    k = 0
    mod = 0

    # TODO(JL): This can and should be done much more efficiently, using
    # kind of binary search to find the right interval
    for k in range(n_c):
        if k == j:
            continue

        # adjust C_start and C_end since we treat current cluster as variable
        mod = cluster_size if k > j else 0

        # check upper end of cluster
        hi_start = cluster_ptr[k] - mod
        hi_end = cluster_ptr[k] + cluster_size - mod

        # check lower end of cluster
        lo_start = cluster_ptr[k + 1] - mod
        lo_end = cluster_ptr[k + 1] + cluster_size - mod

        lo = sum(lambdas[lo_start:lo_end])
        hi = sum(lambdas[hi_start:hi_end])

        new_cluster_ind = k - 1 if k > j else k

        if abs(x) > hi + abs(c[k]):
            # we must be between clusters
            # return np.sign(x) * (np.abs(x) - hi)
            return x - np.sign(x)*hi, new_cluster_ind
        elif abs(x) >= lo + abs(c[k]):
            # we are in a cluster
            return np.sign(x) * abs(c[k]), new_cluster_ind

    new_cluster_ind = k - 1 if k > j else k

    # return np.sign(x) * (np.abs(x) - lo)
    return x - np.sign(x) * lo, new_cluster_ind


@njit
def slope_threshold_opti(x, lambdas, cluster_indices, cluster_ptr, c, n_c, j):
    cluster_size = cluster_ptr[j+1] - cluster_ptr[j]
    zero_lambda_sum = np.sum(lambdas[::-1][0:cluster_size])

    if abs(x) < zero_lambda_sum:
        return 0.0, n_c - 1

    # check which direction we need to search
    up_direction = sum(
        lambdas[cluster_ptr[j]:cluster_ptr[j+1]]) > np.abs(x) - np.abs(c[j])

    if up_direction:
        idx_c = np.arange(j+1, n_c)
        start = cluster_size
        end = 0
    else:
        if j == 0:
            idx_c = np.array([0])
        else:
            idx_c = np.arange(0, j)
        start = 0
        end = cluster_size

    # check upper end of cluste
    hi_start = cluster_ptr[idx_c[0]] - start
    hi_end = cluster_ptr[idx_c[0]] + end
    hi = sum(lambdas[hi_start:hi_end])

    k = 0

    for k in idx_c:

        # check lower end of cluster
        lo_start = cluster_ptr[k+1] - start
        lo_end = cluster_ptr[k+1] + end
        lo = sum(lambdas[lo_start:lo_end])
        new_cluster_ind = k - 1 if up_direction else k + 1

        if abs(x) > hi + abs(c[k]):
            # we must be between clusters
            return x - np.sign(x)*hi, new_cluster_ind

        elif abs(x) >= lo + abs(c[k]):
            # we are in a cluster
            return np.sign(x) * abs(c[k]), new_cluster_ind

        # replace lower interval by higher before next iteration
        hi = lo

    new_cluster_ind = k - 1 if up_direction else k + 1

    return x - np.sign(x) * lo, new_cluster_ind
