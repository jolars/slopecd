from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sparse
from numba import njit
from numpy.linalg import norm
from scipy import stats
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


def lambda_sequence(X, y, fit_intercept, reg=0.1, q=0.1):
    """Generates the BH-type lambda sequence"""
    n, p = X.shape

    randnorm = stats.norm(loc=0, scale=1)
    lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
    lambda_max = dual_norm_slope(X, (y - np.mean(y) * fit_intercept) / n, lambdas)

    return lambda_max * lambdas * reg


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


@njit
def prox_slope2(beta, lambdas):
    """Compute the sorted L1 proximal operator

    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    beta_sign = np.sign(beta)
    beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    beta *= beta_sign

    return beta


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
    else:
        return np.hstack((np.ones((n, 1)), X))


class ConvergenceMonitor:
    def __init__(
        self, X, y, lambdas, tol, gap_freq, max_time, verbose, intercept_column
    ):
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.tol = tol
        self.gap_freq = gap_freq
        self.max_time = max_time
        self.verbose = verbose
        self.intercept_column = intercept_column

        # store gaps, primals, duals, times
        self.gaps, self.primals, self.times = [], [], []

        # start timer
        self.time_start = timer()

        # initialize with null solution
        self.primals.append(norm(y) ** 2 / (2 * X.shape[0]))
        self.gaps.append(self.primals[0])
        self.times.append(0.0)

    def check_convergence(self, w, intercept, epoch):
        n_samples = self.X.shape[0]

        times_up = timer() - self.time_start > self.max_time

        if epoch % self.gap_freq == 0 or times_up:
            if self.intercept_column:
                residual = self.y - self.X @ np.hstack((intercept, w))
                theta = residual / n_samples
                theta /= max(1, dual_norm_slope(self.X[:, 1:], theta, self.lambdas))
            else:
                residual = self.y - self.X @ w - intercept
                theta = residual / n_samples
                theta /= max(1, dual_norm_slope(self.X, theta, self.lambdas))

            primal = norm(residual) ** 2 / (2 * n_samples) + np.sum(
                self.lambdas * np.sort(np.abs(w))[::-1]
            )
            dual = (norm(self.y) ** 2 - norm(self.y - theta * n_samples) ** 2) / (
                2 * n_samples
            )
            gap = primal - dual

            self.primals.append(primal)
            self.gaps.append(gap)

            self.times.append(timer() - self.time_start)

            if self.verbose:
                print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")

            if gap < self.tol or times_up:
                return True
            else:
                return False

    def get_results(self):
        return self.primals, self.gaps, self.times
