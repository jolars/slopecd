import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.cd_utils import compute_grad_hess_sumX
from slope.solvers import prox_grad
from slope.utils import ST, ConvergenceMonitor, get_clusters


@njit
def pure_cd_epoch(w, X, R, alphas, lc):
    n_samples, n_features = X.shape
    for j in range(n_features):
        old = w[j]
        X_j = X[:, j].copy()  # need to do this to have a contiguous array in numba
        w[j] = ST(w[j] + X_j @ R / (lc[j] * n_samples), alphas[j] / lc[j])
        if w[j] != old:
            R += (old - w[j]) * X_j


@njit
def pure_cd_epoch_sparse(
    w, X_data, X_indices, X_indptr, R, alphas, cluster_indices, cluster_ptr, sign_w, c
):
    n_samples = len(R)
    for j in range(len(cluster_ptr) - 1):
        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        w_old = w[j]

        grad, L_j, X_sum_vals, X_sum_inds = compute_grad_hess_sumX(
            R, X_data, X_indices, X_indptr, sign_w[cluster], cluster, n_samples
        )

        x = w_old - grad / (L_j * n_samples)
        w[j] = ST(x, alphas[cluster_ptr[j] : cluster_ptr[j + 1]].sum() / L_j)

        diff = w_old - w[j]
        for i, ind in enumerate(X_sum_inds):
            R[ind] += diff * X_sum_vals[i]


def oracle_cd(
    X,
    y,
    alphas,
    fit_intercept=True,
    w_star=None,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
):
    """Oracle CD: get solution clusters and run CD on collapsed design."""
    n_samples, n_features = X.shape
    if w_star is None:
        w_star = prox_grad(
            X,
            y,
            alphas,
            fit_intercept=fit_intercept,
            max_epochs=10000,
            fista=True,
            tol=1e-10,
        )[0]
    clusters, cluster_ptr, unique = get_clusters(w_star)
    n_clusters = len(cluster_ptr) - 1
    is_X_sparse = sparse.issparse(X)
    # create collapsed design. Beware, we ignore the last cluster, but only
    # if it is 0 valued
    if w_star[clusters[-1]] == 0:
        clusters = clusters[0 : cluster_ptr[-2]]
        cluster_ptr = cluster_ptr[0:-1]
        n_clusters -= 1

    X_reduced = np.zeros([n_samples, n_clusters])
    alphas_reduced = np.zeros(n_clusters)

    if not is_X_sparse:
        for j in range(n_clusters):
            cluster = clusters[cluster_ptr[j] : cluster_ptr[j + 1]]
            X_reduced[:, j] = (X[:, cluster] * np.sign(w_star[cluster])).sum(axis=1)
            alphas_reduced[j] = alphas[cluster_ptr[j] : cluster_ptr[j + 1]].sum()
    # run CD on it:
    w = np.zeros(n_features)
    w_reduced = np.zeros(n_clusters)
    intercept = 0.0
    R = y.copy()

    monitor = ConvergenceMonitor(X, y, alphas, tol, gap_freq, max_time, verbose, False)

    lc = norm(X_reduced, axis=0) ** 2 / n_samples

    for epoch in range(max_epochs):
        if is_X_sparse:
            pure_cd_epoch_sparse(
                w_reduced,
                X.data,
                X.indices,
                X.indptr,
                R,
                alphas,
                clusters,
                cluster_ptr,
                np.sign(w_star),
                unique,
            )
        else:
            pure_cd_epoch(w_reduced, X_reduced, R, alphas_reduced, lc)
        for j in range(n_clusters):
            cluster = clusters[cluster_ptr[j] : cluster_ptr[j + 1]]
            w[cluster] = w_reduced[j] * np.sign(w_star[cluster])

        if fit_intercept:
            intercept_update = np.mean(R)
            R -= intercept_update
            intercept += intercept_update

        converged = monitor.check_convergence(w, intercept, epoch)

        if converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times
