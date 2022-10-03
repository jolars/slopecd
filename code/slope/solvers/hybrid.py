import warnings

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
    XTX,
    R,
    alphas,
    cluster_indices,
    cluster_ptr,
    cluster_perm,
    c,
    n_c,
    cluster_updates,
    update_zero_cluster,
    previously_active,
):
    n_samples = X.shape[0]

    j = 0
    while j < n_c:
        k = cluster_perm[j]
        c_old = c[k]

        if c_old == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c_old != 0 else np.ones(len(cluster))

        if len(cluster) == 1:
            ind = cluster[0]
            sum_X = np.ravel(X[:, ind] * sign_w[0])
            if not previously_active[ind]:
                XTX[ind] = (X[:, ind] @ X[:, ind]) / n_samples
                previously_active[ind] = True
            L_j = XTX[ind]
        else:
            sum_X = np.ravel(X[:, cluster] @ sign_w)
            L_j = (sum_X.T @ sum_X) / n_samples

        x = c_old + sum_X.T @ R / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(
            x, alphas / L_j, cluster_ptr, cluster_perm, c, n_c, j
        )

        c_new = abs(beta_tilde)
        w[cluster] = beta_tilde * sign_w

        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X

        if cluster_updates:
            ind_old = j
            n_c = update_cluster(
                c,
                cluster_ptr,
                cluster_indices,
                cluster_perm,
                n_c,
                c_new,
                c_old,
                ind_old,
                ind_new,
            )
        else:
            c[k] = c_new

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
    cluster_perm,
    c,
    n_c,
    cluster_updates,
    update_zero_cluster,
):
    n_samples = len(R)

    j = 0
    while j < n_c:
        k = cluster_perm[j]
        c_old = c[k]

        if c_old == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c_old != 0 else np.ones(len(cluster))

        grad, L_j, new_vals, new_rows = compute_grad_hess_sumX(
            R, X_data, X_indices, X_indptr, sign_w, cluster, n_samples
        )

        x = c_old - grad / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(
            x, alphas / L_j, cluster_ptr, cluster_perm, c, n_c, j
        )

        c_new = abs(beta_tilde)
        w[cluster] = beta_tilde * sign_w

        diff = c_old - beta_tilde
        if diff != 0:
            for i, ind in enumerate(new_rows):
                R[ind] += diff * new_vals[i]

        if cluster_updates:
            ind_old = j
            n_c = update_cluster(
                c,
                cluster_ptr,
                cluster_indices,
                cluster_perm,
                n_c,
                c_new,
                c_old,
                ind_old,
                ind_new,
            )
        else:
            c[k] = c_new

        j += 1

    return n_c


def hybrid_cd(
    X,
    y,
    alphas,
    w_start=None,
    intercept_start=None,
    fit_intercept=True,
    cluster_updates=True,
    update_zero_cluster=False,
    pgd_freq=5,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
    callback=None,
):
    is_X_sparse = sparse.issparse(X)
    n_samples, n_features = X.shape
    R = y.copy()

    w = np.zeros(n_features) if w_start is None else w_start
    intercept = 0.0 if intercept_start is None else intercept_start

    n_clusters = []
    monitor = ConvergenceMonitor(
        X, y, alphas, tol, gap_freq, max_time, verbose, intercept_column=False
    )

    XTX = np.empty(n_features, dtype=np.float64)

    previously_active = np.zeros(n_features, dtype=bool)

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
    else:
        L = norm(X, ord=2) ** 2 / n_samples

    c, cluster_ptr, cluster_indices, cluster_perm, n_c = get_clusters(w)

    epoch = 0
    if callback is not None:
        proceed = callback(np.hstack((intercept, w)))
    else:
        proceed = True
    while proceed:
        if epoch % pgd_freq == 0:
            s_old = np.sign(w)
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            R[:] = y - X @ w
            if fit_intercept:
                intercept = np.mean(R)
                R -= intercept

            c, cluster_ptr, cluster_indices, cluster_perm, n_c = get_clusters(w)

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
                    cluster_perm,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                )
            else:
                n_c = block_cd_epoch(
                    w,
                    X,
                    XTX,
                    R,
                    alphas,
                    cluster_indices,
                    cluster_ptr,
                    cluster_perm,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                    previously_active,
                )

            if fit_intercept:
                intercept_update = np.mean(R)
                R -= intercept_update
                intercept += intercept_update

        epoch += 1
        if callback is None:
            proceed = epoch < max_epochs
        else:
            proceed = callback(np.hstack((intercept, w)))
        converged = monitor.check_convergence(w, intercept, epoch)
        n_clusters.append(n_c)
        if converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times, n_clusters
