import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
from scipy import stats

from slope.data import get_data
from slope.solvers import admm, hybrid_cd, newt_alm, oracle_cd, prox_grad
from slope.utils import lambda_sequence, preprocess

dataset = "bcTCGA"
X, y = get_data(dataset)

X = preprocess(X)
y = y - np.mean(y)

n, p = X.shape
q = 0.1
fit_intercept = True
reg = 0.1582866

lambdas = lambda_sequence(X, y, fit_intercept, reg, q)

max_epochs = 10000
max_time = 100
verbose = True
tol = 1e-6

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd, _ = hybrid_cd(
    X,
    y,
    lambdas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=verbose,
    tol=tol,
    max_time=max_time,
)

time_cd[-1]
