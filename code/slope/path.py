import numpy as np

from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence


def fit_path(
    X,
    y,
    regs=None,
    path_length=100,
    verbose=False,
    fit_intercept=True,
    q=0.1,
    solver_args={},
):
    n, p = X.shape

    nnz_max = p + 1

    if regs is None:
        auto_reg = True
        reg_min_ratio = 1e-2 if n < n else 1e-4

        regs = np.geomspace(1, reg_min_ratio, 100)
    else:
        auto_reg = False
        path_length = len(regs)

    lambdas = lambda_sequence(X, y, fit_intercept, 1, q)

    null_dev = 0.5 * np.linalg.norm(y - np.mean(y) * fit_intercept) ** 2
    dev_prev = null_dev

    betas = np.zeros((p, path_length))
    intercepts = np.zeros(path_length)

    beta = np.zeros(p)
    intercept = np.mean(y) if fit_intercept else 0.0

    i = 0
    while i < path_length:
        beta, intercept = hybrid_cd(
            X,
            y,
            lambdas * regs[i],
            w_start=beta,
            intercept_start=intercept,
            fit_intercept=fit_intercept,
            **solver_args,
        )[:2]

        betas[:, i] = beta.copy()
        intercepts[i] = intercept.copy()

        dev = 0.5 * np.linalg.norm(y - X @ beta - intercept) ** 2
        dev_ratio = 1 - dev / null_dev
        dev_change = 1 - dev / dev_prev
        nnz = np.sum(np.unique(np.abs(beta)) != 0)

        if verbose:
            print(
                f"step: {i + 1}, reg: {regs[i]:.4f}, dev_ratio: {dev_ratio:.3f}, dev_change: {dev_change:.5f}"
            )

        if auto_reg:
            # path stopping rules taken from glmnet, with the exception of
            # the ever active rule, which does not make sense for SLOPE
            if nnz >= nnz_max or dev_ratio > 0.999 or (i > 0 and dev_change < 1e-5):
                break

        dev_prev = dev
        i += 1

    # strip down results if path was stopped early
    regs = np.delete(regs, range(i, path_length))
    betas = np.delete(betas, range(i, path_length), 1)
    intercepts = np.delete(intercepts, range(i, path_length))

    return betas, intercepts, regs
