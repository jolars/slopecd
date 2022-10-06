import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy import sparse

from slope.cd_utils import sparse_dot_product
from slope.solvers import prox_grad
from slope.utils import ST, ConvergenceMonitor, get_clusters


@njit
def compute_reduced_alphas(alphas, clusters, cluster_ptr):
    n_clusters = len(cluster_ptr) - 1
    alphas_reduced = np.zeros(n_clusters)
    for j in range(n_clusters):
        cluster = clusters[cluster_ptr[j] : cluster_ptr[j + 1]]
        alphas_reduced[j] = np.sum(alphas[cluster_ptr[j] : cluster_ptr[j + 1]])

    return alphas_reduced


@njit
def compute_reduced_X(X, s, clusters, cluster_ptr):
    n_clusters = len(cluster_ptr) - 1
    X_reduced = np.zeros((X.shape[0], n_clusters))
    for j in range(n_clusters):
        cluster = clusters[cluster_ptr[j] : cluster_ptr[j + 1]]
        X_reduced[:, j] = X[:, cluster] @ s[cluster]

    return X_reduced


@njit
def compute_reduced_X_sparse(X_indices, X_indptr, X_data, s, clusters, cluster_ptr):
    X_reduced_row = []
    X_reduced_col = []
    X_reduced_val = []

    n_clusters = len(cluster_ptr) - 1

    for j in range(n_clusters):
        cluster = clusters[cluster_ptr[j] : cluster_ptr[j + 1]]
        for k in cluster:
            start, end = X_indptr[k : k + 2]
            for ind in range(start, end):
                if s[k] != 0:
                    row_ind = X_indices[ind]
                    v = s[k] * X_data[ind]

                    X_reduced_row.append(row_ind)
                    X_reduced_col.append(j)
                    X_reduced_val.append(v)

    return X_reduced_row, X_reduced_col, X_reduced_val


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
    w,
    X_data,
    X_indices,
    X_indptr,
    R,
    alphas,
    lc,
):
    n_samples = len(R)
    n_features = len(w)

    for j in range(n_features):
        old = w[j]

        start, end = X_indptr[j : j + 2]
        X_data_j = X_data[start:end]
        X_indices_j = X_indices[start:end]

        dir = sparse_dot_product(R, X_data_j, X_indices_j)
        w[j] = ST(w[j] + dir / (lc[j] * n_samples), alphas[j] / lc[j])

        diff = old - w[j]

        if diff != 0:
            for i, ind in enumerate(X_indices_j):
                R[ind] += diff * X_data_j[i]


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
    callback=None,
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
        clusters = clusters[: cluster_ptr[-2]]
        cluster_ptr = cluster_ptr[:-1]
        n_clusters -= 1

    monitor = ConvergenceMonitor(X, y, alphas, tol, gap_freq, max_time, verbose, False)

    s = np.sign(w_star)

    if is_X_sparse:
        X_reduced_row, X_reduced_col, X_reduced_val = compute_reduced_X_sparse(
            X.indices, X.indptr, X.data, s, clusters, cluster_ptr
        )
        X_reduced = sparse.coo_matrix(
            (X_reduced_val, (X_reduced_row, X_reduced_col)), dtype=np.float64
        ).tocsc()
    else:
        X_reduced = compute_reduced_X(X, s, clusters, cluster_ptr)

    alphas_reduced = compute_reduced_alphas(alphas, clusters, cluster_ptr)

    # run CD on it:
    w = np.zeros(n_features)
    w_reduced = np.zeros(n_clusters)
    intercept = 0.0
    R = y.copy()

    if is_X_sparse:
        lc = sparse.linalg.norm(X_reduced, axis=0) ** 2 / n_samples
    else:
        lc = norm(X_reduced, 2, axis=0) ** 2 / n_samples

    epoch = 0
    if callback is not None:
        proceed = callback(np.hstack((intercept, w)))
    else:
        proceed = True
    while proceed:
        if is_X_sparse:
            pure_cd_epoch_sparse(
                w_reduced,
                X_reduced.data,
                X_reduced.indices,
                X_reduced.indptr,
                R,
                alphas_reduced,
                lc,
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
        epoch += 1
        if callback is None:
            proceed = epoch < max_epochs
        else:
            proceed = callback(np.hstack((intercept, w)))
        if converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times
