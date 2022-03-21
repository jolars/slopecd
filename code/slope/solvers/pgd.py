import numpy as np
from numpy.linalg import norm
from slope.utils import dual_norm_slope, prox_slope
from scipy import sparse


def prox_grad(
        X, y, alphas, fista=False, max_epochs=100, tol=1e-10, gap_freq=1,
        verbose=True):
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)
    # FISTA parameters:
    z = w.copy()
    t = 1

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
    else:
        L = norm(X, ord=2)**2 / n_samples

    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])
    for it in range(max_epochs):
        R[:] = y - X @ z
        w_new = prox_slope(z + (X.T @ R) / (L * n_samples), alphas / L)

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

            if verbose:
                print(f"Epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")
            if gap < tol:
                break
    return w, E, gaps, theta
