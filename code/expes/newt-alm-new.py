import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as scio
import scipy.sparse
from benchopt.datasets.simulated import make_correlated_data
from matplotlib.cm import get_cmap
from scipy import sparse, stats

from slope.data import get_data
from slope.solvers import hybrid_cd, newt_alm, newt_alm2, prox_grad
from slope.utils import dual_norm_slope, lambda_sequence

cm = get_cmap("tab10")

n = 10
p = 100
rho = 0.3
seed = 14

dataset = "Rhee2006"
if dataset == "simulated":
    X, y, _ = make_correlated_data(
        n_samples=n, n_features=p, random_state=seed, rho=rho
    )
else:
    X, y = get_data(dataset)

n, p = X.shape

A = X.copy()
b = np.expand_dims(y, -1)

randnorm = stats.norm(loc=0, scale=1)
q = 0.05
reg = 0.01
fit_intercept = True

max_epochs = 10_00
gap_freq = 10
tol = 1e-8
verbose = True
solver = "standard"

lambdas = lambda_sequence(X, y, fit_intercept, reg, q)

# w_newt2, intercept_newt2, primals_newt2, gaps_newt2, times_newt2 = newt_alm2(
#     X,
#     y,
#     lambdas,
#     fit_intercept=fit_intercept,
#     preposs=False,
#     gap_freq=1,
#     max_epochs=max_epochs,
#     tol=tol,
#     verbose=verbose,
# )

w_cd, intercept_primals, primals_cd, gaps_cd, times_cd = hybrid_cd(
    X,
    y,
    lambdas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    tol=tol,
    verbose=verbose,
    cluster_updates=True,
)

w_newt, intercept_newt, primals_newt, gaps_newt, times_newt = newt_alm(
    X,
    y,
    lambdas,
    fit_intercept=fit_intercept,
    gap_freq=1,
    max_epochs=max_epochs,
    solver=solver,
    tol=tol,
    verbose=verbose,
    local_param={"epsilon": 1.0, "delta": 1.0, "delta_prime": 1.0, "sigma": 0.001}
)

title = f"{dataset}, n: {n}, p: {p}, q: {q}, reg: {reg}"
if dataset == "simulated":
    title += f", rho: {rho}"

plt.clf()

plt.title(title)

primals_newt = np.array(primals_newt)
# primals_newt2 = np.array(primals_newt2)
primals_cd = np.array(primals_cd)

p_star = min(
    np.min(primals_newt),
    # np.min(primals_newt2),
    np.min(primals_cd)
)

plt.semilogy(times_newt, primals_newt - p_star, c=cm(1), label="primal subopt NEWT-ALM")
# plt.semilogy(
#     times_newt2, primals_newt2 - p_star, c=cm(2), label="primal subopt NEWT-ALM (new)"
# )
plt.semilogy(times_cd, primals_cd - p_star, c=cm(3), label="primal subopt cd")

plt.legend()

plt.show(block=False)
