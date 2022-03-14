from ensurepip import version
import numpy as np
from numba import njit
from numpy.linalg import norm
from slope.utils import ST, dual_norm_slope, prox_slope
from slope.utils import slope_threshold, get_clusters


@njit
def do_cd_epochs(n_cd, w, X, R, alphas, lc):
    n_samples = X.shape[0]
    for _ in range(n_cd):
        # do CD epochs pretending coefs order is fixed
        order = np.argsort(np.abs(w))[::-1]
        for idx, j in enumerate(order):  # update from big to small
            old = w[j]
            w[j] = ST(w[j] + X[:, j] @ R / (lc[j] * n_samples),
                      alphas[idx] / lc[j])
            if w[j] != old:
                R += (old - w[j]) * X[:, j]


def prox_grad(X, y, alphas, max_iter=100, tol=1e-10, n_cd=0, verbose=True):
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)

    L = norm(X, ord=2)**2 / n_samples
    lc = norm(X, axis=0)**2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])

    for t in range(max_iter):
        w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
        R[:] = y - X @ w

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
            print(f"Iter: {t + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break
    return w, E, gaps, theta


@njit
def block_cd_epoch(w, X, R, alphas, cluster_indices, cluster_ptr, c):
    n_samples = X.shape[0]
    for j in range(len(cluster_ptr)-1):
        A = cluster_indices[cluster_ptr[j]:cluster_ptr[j+1]]
        s = np.sign(w[A])
        s = np.ones(len(s)) if np.all(s == 0) else s
        sum_X = s.T @ X[:, A].T
        L_j = sum_X @ sum_X.T / n_samples
        old = np.abs(w[A][0])
        x = old + (sum_X @ R) / (L_j * n_samples)
        beta_tilde = slope_threshold(
            x, alphas/L_j, cluster_indices, cluster_ptr, c, j)
        c[j] = np.abs(beta_tilde)
        w[A] = beta_tilde * s
        R += (old - beta_tilde) * sum_X.T


def hybrid_cd(X, y, alphas, max_iter=1000, verbose=True,
              tol=1e-3):

    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)

    L = norm(X, ord=2)**2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])

    for t in range(max_iter):
        # This is experimental, it will need to be justified
        if t % 5 == 0:
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            R[:] = y - X @ w
            theta = R / n_samples
            theta /= max(1, dual_norm_slope(X, theta, alphas))
            dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
                (2 * n_samples)
            primal = norm(R) ** 2 / (2 * n_samples) + \
                np.sum(alphas * np.sort(np.abs(w))[::-1])

            E.append(primal)
            gap = primal - dual
            gaps.append(gap)
            cluster_indices, cluster_ptr, c = get_clusters(w)
        else:
            block_cd_epoch(w, X, R, alphas, cluster_indices, cluster_ptr, c)

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
            print(f"Iter: {t + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break

    return w, E, gaps


# @njit
def pure_cd_epoch(w, X, R, alphas, lc):
    n_samples, n_features = X.shape
    for j in range(n_features):
        old = w[j]
        w[j] = ST(w[j] + X[:, j] @ R / (lc[j] * n_samples), alphas[j] / lc[j])
        if w[j] != old:
            R += (old - w[j]) * X[:, j]


def oracle_cd(X, y, alphas, max_iter, tol=1e-10, verbose=False):
    """Oracle CD: get solution clusters and run CD on collapsed design."""
    n_samples, n_features = X.shape
    w_star = prox_grad(X, y, alphas, max_iter=10000, tol=1e-10, n_cd=0)[0]
    clusters, cluster_ptr, unique = get_clusters(w_star)
    n_clusters = len(cluster_ptr) - 1

    # create collapsed design. Beware, we ignore the last cluster, but only
    # if it is 0 valued
    if w_star[clusters[-1]] == 0:
        clusters = clusters[0:cluster_ptr[-2]]
        cluster_ptr = cluster_ptr[0:-1]
        n_clusters -= 1

    X_reduced = np.zeros([n_samples, n_clusters])
    alphas_reduced = np.zeros(n_clusters)

    for j in range(n_clusters):
        cluster = clusters[cluster_ptr[j]:cluster_ptr[j+1]]
        X_reduced[:,
                  j] = (X[:, cluster] * np.sign(w_star[cluster])).sum(axis=1)
        alphas_reduced[j] = alphas[cluster_ptr[j]:cluster_ptr[j+1]].sum()
    # run CD on it:
    w = np.zeros(n_features)
    w_reduced = np.zeros(n_clusters)
    R = y.copy()
    lc = norm(X_reduced, axis=0)**2 / n_samples
    E = []
    gaps = []

    for it in range(max_iter):
        pure_cd_epoch(w_reduced, X_reduced, R, alphas_reduced, lc)

        for j in range(n_clusters):
            cluster = clusters[cluster_ptr[j]:cluster_ptr[j+1]]
            w[cluster] = w_reduced[j] * np.sign(w_star[cluster])

        theta = R / n_samples
        theta /= max(1, dual_norm_slope(X, theta, alphas))

        dual = (norm(y)**2 - norm(y - theta * n_samples)**2) / (2 * n_samples)
        primal = norm(R)**2 / (2 * n_samples) + np.sum(
            alphas * np.sort(np.abs(w))[::-1])

        E.append(primal)
        gap = primal - dual
        gaps.append(gap)

        if verbose:
            print(f"Iter: {it + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break

    return w, E, gaps
