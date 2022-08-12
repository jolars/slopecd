from timeit import default_timer as timer

import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.clusters import get_clusters, update_cluster
from slope.utils import dual_norm_slope, prox_slope, slope_threshold


@njit
def update_reduced_X(
    L_archive,
    X_reduced,
    X,
    s,
    cluster_indices,
    cluster_ptr,
    c,
    n_c,
    cluster_indices_old,
    cluster_ptr_old,
    n_c_old,
    s_old,
):
    n_samples = X.shape[0]

    update = False

    for j in range(n_c):  # don't update zero cluster
        if c[j] == 0:
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]

        if j < n_c_old:
            cluster_old = cluster_indices_old[
                cluster_ptr_old[j] : cluster_ptr_old[j + 1]
            ]

            if not np.array_equal(cluster, cluster_old):
                update = True

            if not update:
                for i in cluster:
                    if s[i] != s_old[i]:
                        update = True
                        break
        else:
            update = True

        if update:
            X_reduced[:, j] = X[:, cluster] @ s[cluster]
            L_archive[j] = (X_reduced[:, j].T @ X_reduced[:, j]) / n_samples


@njit
def block_cd_epoch(
    w,
    X,
    X_reduced,
    R,
    alphas,
    cluster_indices,
    cluster_ptr,
    c,
    n_c,
    cluster_updates,
    update_zero_cluster,
    use_reduced_X,
    L_archive,
):
    n_samples = X.shape[0]

    j = 0
    while j < n_c:
        if c[j] == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c[j] != 0 else np.ones(len(cluster))

        if use_reduced_X:
            sum_X = X_reduced[:, j]
            L_j = L_archive[j]
        else:
            sum_X = X[:, cluster] @ sign_w
            L_j = sum_X.T @ sum_X / n_samples

        c_old = abs(c[j])
        x = c_old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(x, alphas / L_j, cluster_ptr, c, n_c, j)

        w[cluster] = beta_tilde * sign_w

        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X

        if use_reduced_X:
            if np.sign(beta_tilde) == -1:
                X_reduced[:, j] = -X_reduced[:, j]

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
    cluster_updates=False,
    update_zero_cluster=False,
    use_reduced_X=False,
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
    theta = np.zeros(n_samples)

    times = []
    time_start = timer()
    times.append(timer() - time_start)

    update_time = 0.0

    # if use_reduced_X:
    X_reduced = np.zeros(X.shape, np.float64, order="F")
    L_archive = np.empty(n_features, np.float64)

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

    E, gaps = [], []
    E.append(norm(y) ** 2 / (2 * n_samples))
    gaps.append(E[0])

    c, cluster_ptr, cluster_indices, n_c = get_clusters(w)

    for epoch in range(max_epochs):
        # This is experimental, it will need to be justified
        if epoch % pgd_freq == 0:
            s_old = np.sign(w)
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            if fit_intercept:
                intercept = intercept + np.sum(R) / (L * n_samples)
            R[:] = y - X @ w - intercept

            t0 = timer()
            if use_reduced_X:
                cluster_ptr_old = cluster_ptr.copy()
                cluster_indices_old = cluster_indices.copy()
                n_c_old = n_c

                c, cluster_ptr, cluster_indices, n_c = get_clusters(w)

                s = np.sign(w)

                update_reduced_X(
                    L_archive,
                    X_reduced,
                    X,
                    s,
                    cluster_indices,
                    cluster_ptr,
                    c,
                    n_c,
                    cluster_indices_old,
                    cluster_ptr_old,
                    n_c_old,
                    s_old,
                )
            else:
                c, cluster_ptr, cluster_indices, n_c = get_clusters(w)

            update_time += timer() - t0
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
                    X_reduced,
                    R,
                    alphas,
                    cluster_indices,
                    cluster_ptr,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                    use_reduced_X,
                    L_archive,
                )

            if fit_intercept:
                intercept_update = np.sum(R) / n_samples
                R -= intercept_update
                intercept += intercept_update

        times_up = timer() - time_start > max_time

        if epoch % gap_freq == 0 or times_up:
            theta = R / n_samples
            theta /= max(1, dual_norm_slope(X, theta, alphas))
            dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / (2 * n_samples)
            primal = norm(R) ** 2 / (2 * n_samples) + np.sum(
                alphas * np.sort(np.abs(w))[::-1]
            )

            E.append(primal)
            gap = primal - dual
            gaps.append(gap)
            times.append(timer() - time_start)

            if verbose:
                print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")
            if gap < tol or times_up:
                break

    print(f"update_time: {update_time}")

    return w, intercept, E, gaps, times
