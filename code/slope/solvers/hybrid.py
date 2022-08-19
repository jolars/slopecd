import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.cd_utils import compute_grad_hess_sumX
from slope.clusters import get_clusters, update_cluster
from slope.utils import ConvergenceMonitor, prox_slope, slope_threshold


# TODO(jolars): This should be updated along the lines of `compute_grad_hess_sumX`
@njit
def compute_block_scalar_sparse(X_data, X_indices, X_indptr, v, cluster, n_samples):
    scal = np.zeros(n_samples)
    if len(cluster) == 1:
        start, end = X_indptr[cluster[0] : cluster[0] + 2]
        scal[X_indices[start:end]] = v * X_data[start:end]
    else:
        for k, j in enumerate(cluster):
            start, end = X_indptr[j : j + 2]
            for ind in range(start, end):
                scal[X_indices[ind]] += v[k] * X_data[ind]
    return scal


@njit
def update_reduced_X_sparse(
    L_archive,
    X_reduced,
    X_data,
    X_indices,
    X_indptr,
    s,
    cluster_indices,
    cluster_ptr,
    cluster_perm,
    c,
    n_c,
    cluster_indices_old,
    cluster_ptr_old,
    cluster_perm_old,
    n_c_old,
    s_old,
):
    n_samples = X_reduced.shape[0]

    update = False

    # TODO(jolars): We reset the permutations because the clusters are
    # refreshed after a PGD step, but it's probably possible to avoid this.
    X_reduced[:, :n_c_old] = X_reduced[:, cluster_perm_old[:n_c_old]]
    L_archive[:n_c_old] = L_archive[cluster_perm_old[:n_c_old]]

    for j in range(n_c):
        k = cluster_perm[j]
        if c[k] == 0:
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]

        # NOTE(jolars): More fine-grained logic is possible here, but I'm not
        # sure that it would do much of a difference in practice except for
        # heavily clustered fits.
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
            X_reduced[:, k] = compute_block_scalar_sparse(
                X_data, X_indices, X_indptr, s[cluster], cluster, n_samples
            )
            L_archive[k] = (X_reduced[:, k].T @ X_reduced[:, k]) / n_samples


@njit
def update_reduced_X(
    L_archive,
    X_reduced,
    X,
    s,
    cluster_indices,
    cluster_ptr,
    cluster_perm,
    c,
    n_c,
    cluster_indices_old,
    cluster_ptr_old,
    cluster_perm_old,
    n_c_old,
    s_old,
):
    n_samples = X.shape[0]

    update = False

    # TODO(jolars): We reset the permutations because the clusters are
    # refreshed after a PGD step, but it's probably possible to avoid this.
    X_reduced[:, :n_c_old] = X_reduced[:, cluster_perm_old[:n_c_old]]
    L_archive[:n_c_old] = L_archive[cluster_perm_old[:n_c_old]]

    for j in range(n_c):
        k = cluster_perm[j]
        if c[k] == 0:
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]

        # NOTE(jolars): More fine-grained logic is possible here, but I'm not
        # sure that it would do much of a difference in practice except for
        # heavily clustered fits.
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
            X_reduced[:, k] = X[:, cluster] @ s[cluster]
            L_archive[k] = (X_reduced[:, k].T @ X_reduced[:, k]) / n_samples


@njit
def block_cd_epoch(
    w,
    X,
    X_reduced,
    R,
    alphas,
    cluster_indices,
    cluster_ptr,
    cluster_perm,
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
        k = cluster_perm[j]
        c_old = c[k]

        if c_old == 0 and not update_zero_cluster:
            j += 1
            continue

        cluster = cluster_indices[cluster_ptr[j] : cluster_ptr[j + 1]]
        sign_w = np.sign(w[cluster]) if c_old != 0 else np.ones(len(cluster))

        if use_reduced_X:
            # NOTE(jolars): numba cannot determine that the slice is contiguous
            # despite it definitely being so. So we have to make a copy here.
            # See https://github.com/numba/numba/issues/8131.
            sum_X = X_reduced[:, k].copy()
            L_j = L_archive[k]
        else:
            sum_X = X[:, cluster] @ sign_w
            L_j = (sum_X.T @ sum_X) / n_samples

        x = c_old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(
            x, alphas / L_j, cluster_ptr, cluster_perm, c, n_c, j
        )

        c_new = abs(beta_tilde)
        w[cluster] = beta_tilde * sign_w

        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X

        if use_reduced_X:
            if np.sign(beta_tilde) == -1:
                X_reduced[:, k] = -X_reduced[:, k]

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
                X_reduced,
                L_archive,
                use_reduced_X,
            )
        else:
            c[k] = c_new

        j += 1

    return n_c


@njit
def block_cd_epoch_sparse(
    w,
    X_reduced,
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
    use_reduced_X,
    L_archive,
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

        if use_reduced_X:
            # NOTE(jolars): numba cannot determine that the slice is contiguous
            # despite it definitely being so. So we have to make a copy here.
            # See https://github.com/numba/numba/issues/8131.
            sum_X = X_reduced[:, k].copy()
            L_j = L_archive[k]
            grad = -(sum_X @ R)
        else:
            grad, L_j, new_vals, new_rows = compute_grad_hess_sumX(
                R, X_data, X_indices, X_indptr, sign_w, cluster, n_samples
            )

        x = c_old - grad / (L_j * n_samples)
        beta_tilde, ind_new = slope_threshold(
            x, alphas / L_j, cluster_ptr, cluster_perm, c, n_c, j
        )

        c_new = abs(beta_tilde)
        w[cluster] = beta_tilde * sign_w

        if use_reduced_X:
            if c_old != beta_tilde:
                R += (c_old - beta_tilde) * sum_X
            if np.sign(beta_tilde) == -1:
                X_reduced[:, k] = -X_reduced[:, k]
        else:
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
                X_reduced,
                L_archive,
                use_reduced_X,
            )
        else:
            c[k] = c_new

        j += 1

    return n_c


def hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=True,
    cluster_updates=True,
    update_zero_cluster=False,
    use_reduced_X=True,
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

    X_reduced = np.empty(X.shape, np.float64, order="F")
    L_archive = np.empty(n_features, np.float64)

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
    else:
        L = norm(X, ord=2) ** 2 / n_samples

    c, cluster_ptr, cluster_indices, cluster_perm, n_c = get_clusters(w)

    for epoch in range(max_epochs):
        if epoch % pgd_freq == 0:
            s_old = np.sign(w)
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            R[:] = y - X @ w
            if fit_intercept:
                intercept = np.mean(R)
                R -= intercept

            if use_reduced_X:
                cluster_ptr_old = cluster_ptr.copy()
                cluster_indices_old = cluster_indices.copy()
                cluster_perm_old = cluster_perm.copy()
                n_c_old = n_c

                c, cluster_ptr, cluster_indices, cluster_perm, n_c = get_clusters(w)

                s = np.sign(w)

                if is_X_sparse:
                    update_reduced_X_sparse(
                        L_archive,
                        X_reduced,
                        X.data,
                        X.indices,
                        X.indptr,
                        s,
                        cluster_indices,
                        cluster_ptr,
                        cluster_perm,
                        c,
                        n_c,
                        cluster_indices_old,
                        cluster_ptr_old,
                        cluster_perm_old,
                        n_c_old,
                        s_old,
                    )
                else:
                    update_reduced_X(
                        L_archive,
                        X_reduced,
                        X,
                        s,
                        cluster_indices,
                        cluster_ptr,
                        cluster_perm,
                        c,
                        n_c,
                        cluster_indices_old,
                        cluster_ptr_old,
                        cluster_perm_old,
                        n_c_old,
                        s_old,
                    )
            else:
                c, cluster_ptr, cluster_indices, cluster_perm, n_c = get_clusters(w)

        else:
            if is_X_sparse:
                n_c = block_cd_epoch_sparse(
                    w,
                    X_reduced,
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
                    use_reduced_X,
                    L_archive,
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
                    cluster_perm,
                    c,
                    n_c,
                    cluster_updates,
                    update_zero_cluster,
                    use_reduced_X,
                    L_archive,
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
