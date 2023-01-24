import numpy as np

from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence


def setup_path(
    X,
    y,
    fit_intercept=True,
    q=0.1,
    path_length=100,
    tol=1e-6,
    max_epochs=100_000,
    verbose=True,
):
    n, p = X.shape

    lambdas = lambda_sequence(X, y, fit_intercept, 1.0, q)
    lambda_min_ratio = 1e-2 if n < p else 1e-4
    reg = np.geomspace(1.0, lambda_min_ratio, path_length)

    intercept = np.mean(y) if fit_intercept else 0.0
    w = np.zeros(p)

    residual = y - intercept

    null_dev = np.linalg.norm(residual) ** 2
    dev_prev = np.inf

    step = 0
    while step < path_length:
        w, intercept = hybrid_cd(
            X,
            y,
            alphas=lambdas * reg[step],
            w_start=w,
            intercept_start=intercept,
            fit_intercept=fit_intercept,
            tol=tol,
            max_epochs=max_epochs,
        )[:2]

        n_c = len(np.unique(np.abs(w)))
        dev = np.linalg.norm(y - intercept - X @ w) ** 2
        dev_ratio = 1 - dev / null_dev
        dev_change = 1 - dev / dev_prev

        if verbose:
            print(
                f"step: {step + 1}, dev_ratio: {dev_ratio:.4f}, "
                + f"dev_change: {dev_change:.6f} n_clusters: {n_c}"
            )

        if dev_ratio >= 0.999 or n_c > min(n, p) or dev_change < 1e-5:
            break

        dev_prev = dev
        step += 1

    return lambdas, reg[: step + 1]
