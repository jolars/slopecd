from timeit import default_timer as timer

import numpy as np
from numpy.linalg import norm
from scipy import sparse

from slope.utils import dual_norm_slope, prox_slope


def prox_grad(
    X,
    y,
    alphas,
    acceleration=None,
    line_search=False,
    gap_freq=1,
    max_epochs=100,
    max_time=np.Inf,
    tol=1e-10,
    verbose=True,
):
    if acceleration not in [None, "anderson", "fista", "bb"]:
        raise ValueError(
            "`acceleration` must be one of None, 'anderson', and 'fista'"
        )

    if acceleration == "bb" and not line_search:
        raise ValueError("cannot use Barzilai-Borwein  rule with `line_search=False`")

    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)
    theta = np.zeros(n_samples)
    grad = np.empty(n_features)

    # FISTA parameters
    z = w.copy()
    t = 1

    # BB parameters
    gamma = 2.0
    prev_bb = 1.0
    w_old = w
    grad_old = grad

    if acceleration == "anderson":
        K = 5
        last_K_w = np.zeros([K + 1, n_features])
        U = np.zeros([K, n_features])

    time_start = timer()

    L = 1

    if not line_search:
        if sparse.issparse(X):
            L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n_samples
        else:
            L = norm(X, ord=2) ** 2 / n_samples

    # Line search parameter
    eta = 2.0

    times = []
    times.append(timer() - time_start)
    E, gaps = [], []
    E.append(norm(y) ** 2 / (2 * n_samples))
    gaps.append(E[0])
    for it in range(max_epochs):
        R[:] = y - X @ z
        if it > 0 and acceleration == "bb":
            grad_old = grad
        grad = -(X.T @ R) / n_samples

        if line_search:
            if acceleration == "bb" and it > 0:
                delta_w = w - w_old
                delta_grad = grad - grad_old

                # BB step size safe-guarding
                delta_w_grad_dot = delta_w @ delta_grad
                bb1 = (norm(delta_w) ** 2) / delta_w_grad_dot
                bb2 = delta_w_grad_dot / (norm(delta_grad) ** 2)

                bb = bb2 if bb1 < gamma * bb2 else bb1 - bb2 / gamma
                L = 1.0 / bb if bb > 0 else 1.0 / prev_bb
            elif acceleration != "bb":
                L *= 0.9

            f_old = norm(R) ** 2 / (2 * n_samples)

            while True:
                w_new = prox_slope(z - grad / L, alphas / L)
                f = norm(X @ w_new - y) ** 2 / (2 * n_samples)
                d = w_new - z
                q = f_old + d @ grad + 0.5 * L * norm(d) ** 2

                if f <= q:
                    break
                else:
                    L *= eta
        else:
            w_new = prox_slope(z - grad / L, alphas / L)

        if acceleration == "anderson":
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
                    p_obj = norm(y - X @ w_new) ** 2 / (2 * n_samples) + np.sum(
                        alphas * np.sort(np.abs(w_new))[::-1]
                    )
                    p_obj_acc = norm(y - X @ w_acc) ** 2 / (2 * n_samples) + np.sum(
                        alphas * np.sort(np.abs(w_acc))[::-1]
                    )
                    if p_obj_acc < p_obj:
                        w_new = w_acc
                except np.linalg.LinAlgError:
                    if verbose:
                        print("linalg error, skipping anderson update")
        w_old = w

        if acceleration == "fista":
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w = w_new
            t = t_new
        else:
            w = w_new
            z = w

        times_up = timer() - time_start > max_time

        if it % gap_freq == 0 or times_up:
            R[:] = y - X @ w
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
                print(f"Epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")
            if gap < tol or times_up:
                break

    return w, np.array(E), np.array(gaps), np.array(times)
