from timeit import default_timer as timer

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import lsqr

from slope.utils import dual_norm_slope, prox_slope


def admm(
    X,
    y,
    lambdas,
    fit_intercept=True,
    rho=1.0,
    alpha=1.0,
    adaptive_rho=True,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
):
    # parameters
    mu = 10
    tau_incr = 2
    tau_decr = 2

    n, p = X.shape

    if fit_intercept:
        if sparse.issparse(X):
            X = sparse.hstack((sparse.csc_array(np.ones((n, 1))), X))
        else:
            X = np.hstack((np.ones((n, 1)), X))

        p += 1

    r = y.copy()

    w = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    theta = np.zeros(n)

    y_norm_2 = norm(y) ** 2

    times = []
    time_start = timer()
    times.append(timer() - time_start)

    # cache factorizations if dense
    if not sparse.issparse(X):
        if n >= p:
            XtX = X.T @ X
            np.fill_diagonal(XtX, XtX.diagonal() + rho)
            L = cholesky(XtX, lower=True)
        else:
            XXt = (X @ X.T) * (1 / rho)
            np.fill_diagonal(XXt, XXt.diagonal() + 1)
            L = cholesky(XXt, lower=True)

        U = L.T

    Xty = X.T @ y

    primals, gaps = [], []
    primals.append(y_norm_2 / (2 * n))

    gaps.append(primals[0])

    for it in range(max_epochs):
        if sparse.issparse(X):
            res = lsqr(
                sparse.vstack((X, np.sqrt(rho) * sparse.eye(p))),
                np.hstack((y, np.sqrt(rho) * (z - u))),
                x0=w,
            )
            w = res[0]
        else:
            q = Xty + rho * (z - u)

            U = L.T

            if n >= p:
                w = solve_triangular(U, solve_triangular(L, q, lower=True))
            else:
                tmp = solve_triangular(U, solve_triangular(L, X @ q, lower=True))
                w = q / rho - (X.T @ tmp) / (rho**2)

        z_old = z.copy()
        w_hat = alpha * w + (1 - alpha) * z_old

        z = w_hat + u
        z[fit_intercept:] = prox_slope(z[fit_intercept:], lambdas * (n / rho))

        u += w_hat - z

        if adaptive_rho:
            # update rho
            r_norm = norm(w - z)
            s_norm = norm(-rho * (z - z_old))

            rho_old = rho

            if r_norm > mu * s_norm:
                rho *= tau_incr
                u /= tau_incr
            elif s_norm > mu * r_norm:
                rho /= tau_decr
                u *= tau_decr

            if rho_old != rho and not sparse.issparse(X):
                # need to refactorize since rho has changed
                if n >= p:
                    np.fill_diagonal(XtX, XtX.diagonal() + (rho - rho_old))
                    L = cholesky(XtX, lower=True)
                else:
                    np.fill_diagonal(XXt, XXt.diagonal() - 1)
                    XXt *= rho_old / rho
                    np.fill_diagonal(XXt, XXt.diagonal() + 1)
                    L = cholesky(XXt, lower=True)

                U = L.T

        times_up = timer() - time_start > max_time

        if it % gap_freq == 0 or times_up:
            r[:] = y - X @ w
            theta = r / n
            theta /= max(1, dual_norm_slope(X[:, fit_intercept:], theta, lambdas))

            dual = (y_norm_2 - norm(y - theta * n) ** 2) / (2 * n)
            primal = norm(r) ** 2 / (2 * n) + np.sum(
                lambdas * np.sort(np.abs(w[fit_intercept:]))[::-1]
            )

            primals.append(primal)
            gap = primal - dual
            gaps.append(gap)
            times.append(timer() - time_start)

            if verbose:
                print(f"Epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")
            if gap < tol or times_up:
                break

    intercept = w[0] if fit_intercept else 0.0

    return w[:fit_intercept], intercept, primals, gaps, times
