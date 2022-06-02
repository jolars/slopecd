from timeit import default_timer as timer

import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.solvers import prox_grad
from slope.utils import ST, dual_norm_slope, get_clusters


@njit
def compute_block_scalar_sparse(
        X_data, X_indices, X_indptr, v, cluster, n_samples):
    scal = np.zeros(n_samples)
    for k, j in enumerate(cluster):
        start, end = X_indptr[j:j+2]
        for ind in range(start, end):
            scal[X_indices[ind]] += v[k] * X_data[ind]
    return scal


@njit
def pure_cd_epoch(w, X, R, alphas, lc):
    n_samples, n_features = X.shape
    for j in range(n_features):
        old = w[j]
        w[j] = ST(w[j] + X[:, j] @ R / (lc[j] * n_samples), alphas[j] / lc[j])
        if w[j] != old:
            R += (old - w[j]) * X[:, j]


@njit
def pure_cd_epoch_sparse(
        w, X_data, X_indices, X_indptr, R, alphas, cluster_indices,
        cluster_ptr, sign_w, c):
    n_samples = len(R)
    for j in range(len(cluster_ptr)-1):
        cluster = cluster_indices[cluster_ptr[j]:cluster_ptr[j+1]]
        sum_X = compute_block_scalar_sparse(
            X_data, X_indices, X_indptr, sign_w[cluster], cluster, n_samples)
        L_j = sum_X.T @ sum_X / n_samples
        old = np.abs(w[cluster][0])
        x = old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde = ST(
            x, alphas[cluster_ptr[j]:cluster_ptr[j+1]].sum()/L_j)
        w[cluster] = beta_tilde * sign_w[cluster]
        R += (old - beta_tilde) * sum_X


def oracle_cd(X, y, alphas, max_epochs, tol=1e-10, max_time=np.Inf, verbose=False):
    """Oracle CD: get solution clusters and run CD on collapsed design."""
    n_samples, n_features = X.shape
    w_star = prox_grad(X, y, alphas, max_epochs=10000, tol=1e-10)[0]
    clusters, cluster_ptr, unique = get_clusters(w_star)
    n_clusters = len(cluster_ptr) - 1
    is_X_sparse = sparse.issparse(X)
    # create collapsed design. Beware, we ignore the last cluster, but only
    # if it is 0 valued
    if w_star[clusters[-1]] == 0:
        clusters = clusters[0:cluster_ptr[-2]]
        cluster_ptr = cluster_ptr[0:-1]
        n_clusters -= 1

    X_reduced = np.zeros([n_samples, n_clusters])
    alphas_reduced = np.zeros(n_clusters)

    if not is_X_sparse:
        for j in range(n_clusters):
            cluster = clusters[cluster_ptr[j]:cluster_ptr[j+1]]
            X_reduced[:, j] = (
                X[:, cluster] * np.sign(w_star[cluster])).sum(axis=1)
            alphas_reduced[j] = alphas[cluster_ptr[j]:cluster_ptr[j+1]].sum()
    # run CD on it:
    w = np.zeros(n_features)
    w_reduced = np.zeros(n_clusters)
    R = y.copy()

    times = []
    time_start = timer()
    times.append(timer() - time_start)

    lc = norm(X_reduced, axis=0)**2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])

    for epoch in range(max_epochs):
        if is_X_sparse:
            pure_cd_epoch_sparse(
                w, X.data, X.indices, X.indptr, R, alphas, clusters,
                cluster_ptr, np.sign(w_star), unique)
        else:
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
        times.append(timer() - time_start)

        times_up = timer() - time_start > max_time

        if verbose:
            print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol or times_up:
            break

    return w, E, gaps, times
