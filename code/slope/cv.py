import numpy as np
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from slope.path import fit_path
from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence, sl1_norm


def cv(
    X,
    y,
    path_length=100,
    n_folds=10,
    random_state=0,
    verbosity=0,
    fit_intercept=True,
    q=0.1,
    **kwargs
):
    # remove zero-var variables
    X = VarianceThreshold().fit_transform(X)

    n, p = X.shape

    # setup standardizer
    scaler = MaxAbsScaler if sparse.issparse(X) else StandardScaler

    X_full = scaler().fit_transform(X)

    if sparse.issparse(X_full):
        X_full = X_full.tocsc()

    if verbosity >= 1:
        print(f"Fitting full path to obtain path values.")

    betas, intercepts, regs = fit_path(
        X_full,
        y,
        path_length=path_length,
        verbose=verbosity >= 2,
        fit_intercept=fit_intercept,
        q=q,
        solver_args=kwargs
    )

    real_path_length = len(regs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    res = np.zeros((real_path_length, n_folds))

    i = 0
    for train_index, test_index in kf.split(X):
        if verbosity >= 1:
            print(f"fold: {i + 1}/{n_folds}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler_fit = scaler().fit(X_train)
        X_train = scaler_fit.transform(X_train)
        X_test = scaler_fit.transform(X_test)

        if sparse.issparse(X_train):
            X_train = X_train.tocsc()

        betas, intercepts, _ = fit_path(
            X_train,
            y_train,
            regs=regs,
            verbose=verbosity >= 2,
            fit_intercept=fit_intercept,
            q=q,
            solver_args=kwargs
        )

        lambdas = lambda_sequence(X_test, y_test, fit_intercept, 1.0, q)

        # compute primal value across folds
        for j in range(real_path_length):
            res[j, i] = 0.5 * np.linalg.norm(
                y_test - X_test @ betas[:, j] - intercepts[j]
            ) ** 2 + sl1_norm(betas[:, j], lambdas * regs[j])

        i += 1

    return res, regs
