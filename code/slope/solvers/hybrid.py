import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.cd_utils import compute_grad_hess_sumX
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

        c_old = c[j]

        grad, L_j, new_vals, new_rows = compute_grad_hess_sumX(
            R, X_data, X_indices, X_indptr, sign_w, cluster, n_samples
        )

        x = c_old - grad / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(x, alphas / L_j, cluster_ptr, c, n_c, j)

        w[cluster] = beta_tilde * sign_w

        diff = c_old - beta_tilde
        if diff != 0:
            for i, ind in enumerate(new_rows):
                R[ind] += diff * new_vals[i]

        if cluster_updates:
            ind_old = j
            n_c = update_cluster(
                c, cluster_ptr, cluster_indices, n_c, beta_tilde, ind_old, ind_new
            )
        else:
            c[j] = abs(beta_tilde)

        j += 1

    return n_c


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
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
    else:
        L = norm(X, ord=2) ** 2 / n_samples

    for epoch in range(max_epochs):
        # This is experimental, it will need to be justified
        if epoch % pgd_freq == 0:
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            R[:] = y - X @ w
            if fit_intercept:
                intercept = np.mean(R)
                R -= intercept
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
                intercept_update = np.mean(R)
                R -= intercept_update
                intercept += intercept_update

        converged = monitor.check_convergence(w, intercept, epoch)

        if converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times
