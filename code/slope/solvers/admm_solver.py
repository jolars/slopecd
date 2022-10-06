import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import lsqr

from slope.utils import add_intercept_column, prox_slope, ConvergenceMonitor


def admm(
    X,
    y,
    lambdas,
    fit_intercept=True,
    rho=1.0,
    alpha=1.0,
    adaptive_rho=True,
    lsqr_atol=1e-6,
    lsqr_btol=1e-6,
    gap_freq=10,
    tol=1e-6,
    max_epochs=10_000,
    max_time=np.inf,
    verbose=False,
    callback=None
):
    # implementation from https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
    # splitting: min f(w) + g(z) subject to z = w + use *scaled* Lagrangian variable u

    # parameters
    mu = 10
    tau_incr = 2
    tau_decr = 2

    n, p = X.shape

    if fit_intercept:
        X = add_intercept_column(X)
        p += 1

    monitor = ConvergenceMonitor(
        X, y, lambdas, tol, gap_freq, max_time, verbose, intercept_column=fit_intercept
    )

    w = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    do_lsqr = sparse.issparse(X) and min(n, p) > 1000

    # cache factorizations if dense
    if not do_lsqr:
        if n >= p:
            XtX = X.T @ X
            if sparse.issparse(X):
                XtX = XtX.toarray()
            np.fill_diagonal(XtX, XtX.diagonal() + rho)
            L = cholesky(XtX, lower=True)
        else:
            XXt = X @ X.T
            if sparse.issparse(X):
                XXt = XXt.toarray()
            XXt *= 1 / rho
            np.fill_diagonal(XXt, XXt.diagonal() + 1)
            L = cholesky(XXt, lower=True)

        U = L.T

    Xty = X.T @ y

    it = 0
    intercept = w[0] if fit_intercept else 0.0
    if callback is not None:
        proceed = callback(np.hstack((intercept, w[fit_intercept:])))
    else:
        proceed = True
    while proceed:
        if do_lsqr:
            res = lsqr(
                sparse.vstack((X, np.sqrt(rho) * sparse.eye(p))),
                np.hstack((y, np.sqrt(rho) * (z - u))),
                x0=w,
                atol=lsqr_atol,
                btol=lsqr_btol,
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

            if rho_old != rho and not do_lsqr:
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

        intercept = w[0] if fit_intercept else 0.0
        converged = monitor.check_convergence(w[fit_intercept:], intercept, it)

        it += 1
        if callback is None:
            proceed = it < max_epochs
        else:
            proceed = callback(np.hstack((intercept, w[fit_intercept:])))
        if callback is None and converged:
            break
    primals, gaps, times = monitor.get_results()

    intercept = w[0] if fit_intercept else 0.0

    return w[fit_intercept:], intercept, primals, gaps, times
