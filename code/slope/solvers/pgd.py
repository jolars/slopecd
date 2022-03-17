import numpy as np
from numba import njit
from numpy.linalg import norm
from slope.utils import ST, dual_norm_slope, prox_slope
from scipy import sparse


@njit
def do_cd_epochs(n_cd, w, X, R, alphas, lc):
    n_samples = len(R)
    for _ in range(n_cd):
        # do CD epochs pretending coefs order is fixed
        order = np.argsort(np.abs(w))[::-1]
        for idx, j in enumerate(order):  # update from big to small
            old = w[j]
            w[j] = ST(w[j] + X[:, j] @ R / (lc[j] * n_samples),
                      alphas[idx] / lc[j])
            if w[j] != old:
                R += (old - w[j]) * X[:, j]


def do_cd_epochs_sparse(
        n_cd, w, X_data, X_indices, X_indptr, R, alphas, lc):
    n_samples = len(R)
    for _ in range(n_cd):
        # do CD epochs pretending coefs order is fixed
        order = np.argsort(np.abs(w))[::-1]
        for idx, j in enumerate(order):  # update from big to small
            old = w[j]
            scal = 0.
            start, end = X_indptr[j:j+2]
            for ind in range(start, end):
                scal += X_data[ind] * R[X_indices[ind]]
            w[j] = ST(w[j] + scal / (lc[j] * n_samples),
                      alphas[idx] / lc[j])
            diff = old - w[j]
            if diff != 0:
                for ind in range(start, end):
                    R[X_indices[ind]] += diff * X_data[ind]


def prox_grad(X, y, alphas, max_epochs=100, tol=1e-10, n_cd=0, verbose=True):
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2
        L /= n_samples
        lc = np.array((X.multiply(X)).sum(axis=0)).squeeze()
    else:
        L = norm(X, ord=2)**2 / n_samples
        lc = norm(X, axis=0)**2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])
    for t in range(max_epochs):
        w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
        R[:] = y - X @ w
        if n_cd > 0:
            do_cd_epochs(n_cd, w, X, R, alphas, lc)

        theta = R / n_samples
        theta /= max(1, dual_norm_slope(X, theta, alphas))

        dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
            (2 * n_samples)
        primal = norm(R) ** 2 / (2 * n_samples) + \
            np.sum(alphas * np.sort(np.abs(w))[::-1])

        E.append(primal)
        gap = primal - dual
        gaps.append(gap)

        if verbose:
            print(f"Epoch: {t + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break
    return w, E, gaps, theta
