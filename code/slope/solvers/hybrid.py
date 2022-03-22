import numpy as np
from numba import njit
from numpy.linalg import norm
from slope.utils import dual_norm_slope, prox_slope
from slope.utils import slope_threshold, get_clusters
from scipy import sparse
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


@njit
def compute_X_reduced(
        sign_w, X_data, X_indices, X_indptr, cluster_indices, cluster_ptr, c):
    X_data_reduced = list()
    X_indices_reduced = list()
    X_indptr_reduced = list()
    X_indptr_reduced.append(0)
    for idx in range(len(c)):
        cluster = cluster_indices[cluster_ptr[idx]:cluster_ptr[idx+1]]
        count = 0
        if c[idx] == 0:
            X_data_reduced.append(1)
            X_indices_reduced.append(1)
            X_indptr_reduced.append(X_indptr_reduced[-1] + 1)
            break
        for k, j in enumerate(cluster):
            start, end = X_indptr[j:j+2]
            if k == 0:
                X_data_reduced.extend(X_data[start:end] * sign_w[j])
                X_indices_reduced.extend(X_indices[start:end])
                count += (end - start)
                X_indptr_reduced.append(X_indptr_reduced[-1] + count)
            else:
                pt1 = start
                pt2 = X_indptr_reduced[-2]
                while pt1 < end:
                    if X_indices[pt1] > X_indices_reduced[-1]:
                        X_data_reduced.append(X_data[pt1] * sign_w[j])
                        X_indices_reduced.append(X_indices[pt1])
                        pt1 += 1
                        count += 1
                        continue
                    if X_indices[pt1] == X_indices_reduced[pt2]:
                        X_data_reduced[pt2] += X_data[pt1] * sign_w[j]
                        pt1 += 1
                        pt2 += 1

                    elif X_indices[pt1] > X_indices_reduced[pt2]:
                        pt2 += 1

                    else:
                        X_data_reduced.insert(pt2, X_data[pt1] * sign_w[j])
                        X_indices_reduced.insert(pt2, X_indices[pt1])
                        count += 1
                        pt1 += 1
                        pt2 += 1

                X_indptr_reduced[-1] = X_indptr_reduced[-2] + count

    return np.array(X_data_reduced), np.array(X_indices_reduced), \
        np.array(X_indptr_reduced)


@njit
def block_cd_epoch(w, X, R, alphas, cluster_indices, cluster_ptr, c):
    n_samples = X.shape[0]
    for j in range(len(c)):
        if c[j] == 0:
            continue
        cluster = cluster_indices[cluster_ptr[j]:cluster_ptr[j+1]]
        sign_w = np.sign(w[cluster])
        sum_X = X[:, cluster] @ sign_w
        L_j = sum_X.T @ sum_X / n_samples
        c_old = c[j]
        x = c_old + (sum_X.T @ R) / (L_j * n_samples)
        beta_tilde = slope_threshold(
            x, alphas/L_j, cluster_indices, cluster_ptr, c, j)
        c[j] = beta_tilde
        w[cluster] = beta_tilde * sign_w
        if c_old != beta_tilde:
            R += (c_old - beta_tilde) * sum_X


@njit
def block_cd_epoch_sparse(w, X_data_reduced, X_indices_reduced,
                          X_indptr_reduced,
                          R, alphas, cluster_indices, cluster_ptr, c):
    n_samples = len(R)
    for j in range(len(c)):
        if c[j] == 0:
            continue
        start, end = X_indptr_reduced[j:j+2]
        X_j = X_data_reduced[start:end]
        cluster = cluster_indices[cluster_ptr[j]:cluster_ptr[j+1]]
        sign_w = np.sign(w[cluster])
        L_j = np.sum(X_j ** 2) / n_samples
        c_old = c[j]
        x = c_old + (X_j.T @ R[X_indices_reduced[start:end]]) \
            / (L_j * n_samples)
        beta_tilde = slope_threshold(
            x, alphas/L_j, cluster_indices, cluster_ptr, c, j)
        c[j] = beta_tilde
        w[cluster] = beta_tilde * sign_w
        if c_old != beta_tilde:
            R[X_indices_reduced[start:end]] += (c_old - beta_tilde) * X_j


@njit
def compute_block_scalar_sparse(
        X_data, X_indices, X_indptr, v, cluster, n_samples):
    scal = np.zeros(n_samples)
    for k, j in enumerate(cluster):
        start, end = X_indptr[j:j+2]
        for ind in range(start, end):
            scal[X_indices[ind]] += v[k] * X_data[ind]
    return scal


def hybrid_cd(X, y, alphas, max_epochs=1000, verbose=True,
              tol=1e-3):

    is_X_sparse = sparse.issparse(X)
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)

    if is_X_sparse:
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2
        L /= n_samples
    else:
        L = norm(X, ord=2)**2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])

    for epoch in range(max_epochs):
        # This is experimental, it will need to be justified
        if epoch % 5 == 0:
            w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
            R[:] = y - X @ w
            cluster_indices, cluster_ptr, c = get_clusters(w)
            if is_X_sparse:
                X_data_reduced, X_indices_reduced, X_indptr_reduced = \
                    compute_X_reduced(
                        np.sign(w), X.data, X.indices, X.indptr,
                        cluster_indices, cluster_ptr, c)
        else:
            if is_X_sparse:
                block_cd_epoch_sparse(
                    w, X_data_reduced, X_indices_reduced, X_indptr_reduced, R,
                    alphas, cluster_indices, cluster_ptr, c)
            else:
                block_cd_epoch(
                    w, X, R, alphas, cluster_indices, cluster_ptr, c)

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
            print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break

    return w, E, gaps
