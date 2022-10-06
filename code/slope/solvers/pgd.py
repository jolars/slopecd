import numpy as np
from numpy.linalg import norm
from scipy import sparse

from slope.utils import ConvergenceMonitor, prox_slope


def prox_grad(
    X,
    y,
    alphas,
    fit_intercept=True,
    fista=False,
    anderson=False,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
    callback=None
):
    if anderson and fista:
        raise ValueError("anderson=True cannot be combined with fista=True")
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    intercept = 0.0

    # FISTA parameters:
    z = w.copy()
    t = 1

    monitor = ConvergenceMonitor(X, y, alphas, tol, gap_freq, max_time, verbose, False)

    if anderson:
        K = 5
        last_K_w = np.zeros([K + 1, n_features])
        U = np.zeros([K, n_features])

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
    else:
        L = norm(X, ord=2) ** 2 / n_samples

    it = 0
    if callback is not None:
        proceed = callback(np.hstack((intercept, w)))
    else:
        proceed = True
    while proceed:
        w_new = prox_slope(z + (X.T @ R) / (L * n_samples), alphas / L)
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
                    w_acc = np.sum(last_K_w[:-1] * c[:, None], axis=0)
                    p_obj = norm(y - X @ w_new - intercept) ** 2 / (
                        2 * n_samples
                    ) + np.sum(alphas * np.sort(np.abs(w_new))[::-1])
                    p_obj_acc = norm(y - X @ w_acc - intercept) ** 2 / (
                        2 * n_samples
                    ) + np.sum(alphas * np.sort(np.abs(w_acc))[::-1])
                    if p_obj_acc < p_obj:
                        w_new = w_acc
                except np.linalg.LinAlgError:
                    if verbose:
                        print("----------Linalg error")

        if fista:
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w = w_new
            t = t_new
        else:
            w = w_new
            z = w

        R[:] = y - X @ z
        if fit_intercept:
            intercept = np.mean(R)
        R -= intercept

        converged = monitor.check_convergence(w, intercept, it)
        it += 1
        if callback is None:
            proceed = it < max_epochs
        else:
            proceed = callback(np.hstack((intercept, w)))
        if callback is None and converged:
            break

    primals, gaps, times = monitor.get_results()

    return w, intercept, primals, gaps, times
