from scipy import stats
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm

from sklearn.isotonic import isotonic_regression


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


def prox_grad(X, y, alphas, max_iter=100, tol=1e-10, verbose=True):
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)

    L = norm(X, ord=2) ** 2 / n_samples
    E = []
    for t in range(max_iter):
        w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
        R[:] = y - X @ w
        theta = R / n_samples
        theta /= dual_norm_slope(X, theta, alphas)
        dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
            (2 * n_samples)
        primal = norm(R) ** 2 / (2 * n_samples) + \
            np.sum(alphas * np.sort(np.abs(w))[::-1])
        E.append(primal)
        gap = primal - dual
        if verbose:
            print(f"Iter: {t + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break
    return w, E, theta


X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas = 0.1 * \
    randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))
# alphas = np.full(X.shape[1], np.max(np.abs(X.T @ y)) / len(y) / 4)

w, E, theta = prox_grad(X, y, alphas, max_iter=1000)
R = y - X @ w
