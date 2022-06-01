from timeit import default_timer as timer

import numpy as np
from numpy.linalg import norm
from scipy import sparse

from slope.utils import dual_norm_slope, prox_slope


def prox_grad(
    X,
    y,
    alphas,
    fit_intercept=True,
    fista=False,
    max_epochs=100,
    tol=1e-10,
    gap_freq=1,
    anderson=False,
    verbose=True,
):
    if anderson and fista:
        raise ValueError("anderson=True cannot be combined with fista=True")
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    intercept = 0.0
    theta = np.zeros(n_samples)
    # FISTA parameters:
    z = w.copy()
    t = 1

    if anderson:
        K = 5
        last_K_w = np.zeros([K + 1, n_features])
        U = np.zeros([K, n_features])

    times = []
    time_start = timer()
    times.append(timer() - time_start)

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

        L = spectral_norm ** 2 / n_samples

    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])
    for it in range(max_epochs):
        R[:] = y - X @ z - intercept
        w_new = prox_slope(z + (X.T @ R) / (L * n_samples), alphas / L)
        if fit_intercept:
            intercept += np.sum(R) / (L * n_samples)
        if anderson:
            # TODO multiple improvements possible here
            if it < K + 1:
                last_K_w[it] = w_new
            else:
                for k in range(K):
                    last_K_w[k] = last_K_w[k + 1]
                last_K_w[K] = w_new

                for k in range(K):
                    U[k] = last_K_w[k + 1] - last_K_w[k]
                C = np.dot(U, U.T)

                try:
                    coefs = np.linalg.solve(C, np.ones(K))
                    c = coefs / coefs.sum()
                    w_acc = np.sum(last_K_w[:-1] * c[:, None],
                                   axis=0)
                    p_obj = norm(y - X @ w_new - intercept) ** 2 / (2 * n_samples) + \
                        np.sum(alphas * np.sort(np.abs(w_new))[::-1])
                    p_obj_acc = norm(y - X @ w_acc - intercept) ** 2 / (
                        2 * n_samples
                    ) + np.sum(alphas * np.sort(np.abs(w_acc))[::-1])
                    if p_obj_acc < p_obj:
                        w_new = w_acc
                except np.linalg.LinAlgError:
                    if verbose:
                        print("----------Linalg error")

        if fista:
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w = w_new
            t = t_new
        else:
            w = w_new
            z = w

        if it % gap_freq == 0:
            R[:] = y - X @ w - intercept
            theta = R / n_samples
            theta /= max(1, dual_norm_slope(X, theta, alphas))

            dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
                (2 * n_samples)
            primal = norm(R) ** 2 / (2 * n_samples) + \
                np.sum(alphas * np.sort(np.abs(w))[::-1])

            E.append(primal)
            gap = primal - dual
            gaps.append(gap)
            times.append(timer() - time_start)

            if verbose:
                print(f"Epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")
            if gap < tol:
                break
    return w, intercept, E, gaps, times
