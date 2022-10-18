from timeit import default_timer as timer

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
verbose = False
tol = 1e-6

# run once for numba JIT compiling
hybrid_cd(
    X,
    y,
    lambdas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=False,
    tol=1e-3,
    max_time=max_time,
)

t0 = timer()
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

slopecd_time = timer() - t0

print(f"slopecd: {slopecd_time}")

f = open("results/bcTCGA_slopecd.txt", "w")
f.write(str(slopecd_time) + "\n")
f.close()
