from timeit import default_timer as timer

import numpy as np
from numpy.linalg import norm
from scipy import sparse

from slope.utils import dual_norm_slope, prox_slope


def prox_grad(
        X, y, alphas, fista=False, max_epochs=100, tol=1e-10, gap_freq=1,
        anderson=False, line_search=False, verbose=True):
    if anderson and fista:
        raise ValueError("anderson=True cannot be combined with fista=True")
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
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

    L = 1.0

    if not line_search:
        if sparse.issparse(X):
            L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
        else:
            L = norm(X, ord=2)**2 / n_samples
    print("init L", L)
    # Line search parameter
    eta = 2.0

    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])
    for it in range(max_epochs):
        R[:] = y - X @ z
        grad = -(X.T @ R) / n_samples

        if line_search:
            L = 1
            f_old = norm(R) ** 2 / (2 * n_samples)
            while True:
                w_new = prox_slope(z - grad / L, alphas / L)
                f = norm(X @ w_new - y) ** 2 / (2 * n_samples)
                d = w_new - z
                q = f_old + np.dot(d, grad) + 0.5 * L * norm(d)**2
                if q >= f * (1 - 1e-12):
                    break
                else:
                    L *= eta
            print("it", it, L)
        else:
            w_new = prox_slope(z - grad / L, alphas / L)

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
                    p_obj = norm(y - X @ w_new) ** 2 / (2 * n_samples) + \
                        np.sum(alphas * np.sort(np.abs(w_new))[::-1])
                    p_obj_acc = norm(y - X @ w_acc) ** 2 / (2 * n_samples) + \
                        np.sum(alphas * np.sort(np.abs(w_acc))[::-1])
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
            R[:] = y - X @ w
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
    return w, np.array(E), np.array(gaps), np.array(times)
