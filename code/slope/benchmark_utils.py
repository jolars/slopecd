import numpy as np
from sklearn.linear_model import LinearRegression
from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence


# figure out the reg needed for a certain dev_ratio
def get_reg_devratio(
    dev_ratios,
    X,
    y,
    q = 0.1,
    fit_intercept = True,
    tol=0.005,
    verbose = False
):
    n_samples, n_features = X.shape

    dev_ratios = np.asarray(dev_ratios)

    if np.any(dev_ratios <= 0):
        raise ValueError("dev_ratios must strictly positive")

    if np.any(dev_ratios > 0.999):
        raise ValueError("dev_ratios cannot be higher than 0.999")

    if np.all(np.diff(dev_ratios) < 0):
        raise ValueError("dev_ratios must strictly increasing")

    lambda_min_ratio = 1e-2 if n_samples < n_features else 1e-4

    lambdas = lambda_sequence(X, y, fit_intercept, 1, q)

    regs = np.geomspace(1, lambda_min_ratio, 100)

    r = y - np.mean(y) if fit_intercept else y.copy()
    null_dev = 0.5 * np.linalg.norm(r) ** 2

    # if n > p, find R2 (dev ratio of OLS) and take a fraction of that
    if n_samples > n_features:
        if verbose:
            print(f"Fitting OLS to get maximum dev ratio.")
        dev_ratio_full = (
            LinearRegression(fit_intercept=fit_intercept).fit(X, y).score(X, y)
        )
    else:
        dev_ratio_full = 1.0

    dev_ratio_targets = dev_ratios * dev_ratio_full

    if verbose:
        print(f"Maximum dev ratio: {dev_ratio_full:.2f}")
        print(f"Dev ratio targets: {dev_ratio_targets}")

    def f(reg, dev_ratio_target, w_start, intercept_start):
        w, intercept = hybrid_cd(
            X,
            y,
            lambdas * reg,
            w_start=w_start,
            intercept_start=intercept_start,
            tol=1e-4,
            fit_intercept=fit_intercept,
            use_reduced_X=False,
            gap_freq=1
        )[:2]

        dev = 0.5 * np.linalg.norm(y - X @ w - intercept) ** 2
        dev_ratio = 1 - dev / null_dev

        return dev_ratio - dev_ratio_target, w, intercept

    if verbose:
        print(f"Fitting path to locate region for regs.")

    los = []
    his = []
    it_max = 1000

    w = np.zeros(n_features)
    intercept = 0.0

    regs_out = []

    j = 0
    i = 0
    if verbose:
        print(f"  Target 1: {dev_ratio_targets[0]:.3f}")
    while i < len(regs):
        f_i, w, intercept = f(regs[i], dev_ratio_targets[j], w, intercept)
        if verbose:
            print(f"    step: {i}, reg: {regs[i]:.3f}, dev_ratio: {f_i + dev_ratio_targets[j]:.3f}")
        if f_i >= 0:
            w_old = w.copy() # store for next target

            a = regs[i]
            b = regs[i - 1] if i > 0 else 1.0

            if verbose:
                print(f"    Bisect to obtain reg value within tol: {tol}.")
            for it in range(it_max):
                c = (a + b) / 2

                f_c, w, intercept = f(c, dev_ratio_targets[j], w, intercept)

                if verbose:
                    print(f"      Dev ratio: {f_c + dev_ratio_targets[j]:.3f}, diff: {f_c:.3f}")

                if abs(f_c) <= tol:
                    regs_out.append(c)
                    break

                f_a, _, _ = f(a, dev_ratio_targets[j], w, intercept)

                if np.sign(f_c) == np.sign(f_a):
                    a = c
                else:
                    b = c

                if it == it_max - 1:
                    raise ValueError("Bisection did not converge.")

            j += 1
            w = w_old

            if j == len(dev_ratio_targets):
                break

            if verbose:
                print(f"  Target {j + 1}: {dev_ratio_targets[j]:.3f}")
        else:
            i += 1

    return regs_out, dev_ratio_targets
