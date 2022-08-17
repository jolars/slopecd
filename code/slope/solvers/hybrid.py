import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.clusters import get_clusters, update_cluster
from slope.utils import ConvergenceMonitor, prox_slope, slope_threshold


@njit
def block_cd_epoch(
    w,
    X,
    R,
    alphas,
    cluster_indices,
    cluster_ptr,
    c,
    n_c,
    cluster_updates,
    update_zero_cluster,
):
    n_samples = X.shape[0]

    j = 0
    while j < n_c:
        if c[j] == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c[j] != 0 else np.ones(len(cluster))
        sum_X = X[:, cluster] @ sign_w
        L_j = sum_X.T @ sum_X / n_samples
        c_old = abs(c[j])
        x = c_old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(x, alphas / L_j, cluster_ptr, c, n_c, j)

        w[cluster] = beta_tilde * sign_w
        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X

        if cluster_updates:
            ind_old = j
            n_c = update_cluster(
                c, cluster_ptr, cluster_indices, n_c, abs(beta_tilde), ind_old, ind_new
            )
        else:
            c[j] = beta_tilde

        j += 1

    return n_c


@njit
def block_cd_epoch_sparse(
    w,
    X_data,
    X_indices,
    X_indptr,
    R,
    alphas,
    cluster_indices,
    cluster_ptr,
    c,
    n_c,
    cluster_updates,
    update_zero_cluster,
):
    n_samples = len(R)

    j = 0
    while j < n_c:
        if c[j] == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c[j] != 0 else np.ones(len(cluster))
        sum_X = compute_block_scalar_sparse(
            X_data, X_indices, X_indptr, sign_w, cluster, n_samples
        )
        L_j = sum_X.T @ sum_X / n_samples
        c_old = abs(c[j])
        x = c_old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(x, alphas / L_j, cluster_ptr, c, n_c, j)
        w[cluster] = beta_tilde * sign_w
        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X

        if cluster_updates:
            ind_old = j
            n_c = update_cluster(
                c, cluster_ptr, cluster_indices, n_c, beta_tilde, ind_old, ind_new
            )
        else:
            c[j] = beta_tilde

        j += 1

    return n_c


@njit
def compute_block_scalar_sparse(X_data, X_indices, X_indptr, v, cluster, n_samples):
    scal = np.zeros(n_samples)
    for k, j in enumerate(cluster):
        start, end = X_indptr[j : j + 2]
        for ind in range(start, end):
            scal[X_indices[ind]] += v[k] * X_data[ind]
    return scal


def hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=True,
    cluster_updates=True,
    update_zero_cluster=False,
    pgd_freq=5,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
):
    is_X_sparse = sparse.issparse(X)
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    intercept = 0.0

    monitor = ConvergenceMonitor(
        X, y, alphas, tol, gap_freq, max_time, verbose, intercept_column=False
    )

    if sparse.issparse(X):
        if fit_intercept:
            # TODO: consider if it's possible to avoid creating this
            # temporary design matrix with a column of ones
            ones_col = sparse.csc_array(np.ones((n_samples, 1)))
            decomp = sparse.linalg.svds(sparse.hstack((ones_col, X)), k=1)
        else:
            decomp = sparse.linalg.svds(X, k=1)

        L = decomp[1][0] ** 2 / n_samples
    else:
        if fit_intercept:
            spectral_norm = norm(np.hstack((np.ones((n_samples, 1)), X)), ord=2)
        else:
            spectral_norm = norm(X, ord=2)

        L = spectral_norm**2 / n_samples

    for epoch in range(max_epochs):
        # This is experimental, it will need to be justified
        if epoch % pgd_freq == 0:
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            if fit_intercept:
                intercept = intercept + np.sum(R) / (L * n_samples)
            R[:] = y - X @ w - intercept
            c, cluster_ptr, cluster_indices, n_c = get_clusters(w)
        else:
            if is_X_sparse:
                n_c = block_cd_epoch_sparse(
                    w,
                    X.data,
                    X.indices,
                    X.indptr,
                    R,
                    alphas,
                    cluster_indices,
                    cluster_ptr,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                )
            else:
                n_c = block_cd_epoch(
                    w,
                    X,
                    R,
                    alphas,
                    cluster_indices,
                    cluster_ptr,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                )

            if fit_intercept:
                intercept_update = np.sum(R) / n_samples
                R -= intercept_update
                intercept += intercept_update

        converged = monitor.check_convergence(w, intercept, epoch)

        if converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times
