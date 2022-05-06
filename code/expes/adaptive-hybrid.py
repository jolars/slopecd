import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats
from scipy.sparse import csc_matrix

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

rho = 0.2
n = 100
p = 20000
reg = 0.1

X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=rho)

n, p = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.1

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg
max_epochs = 5000
tol = 1e-8
n_it = 1
verbose = False
cluster_updates = True

beta, primals, gaps, time = hybrid_cd(
    X,
    y,
    alphas,
    adaptive=False,
    max_epochs=max_epochs,
    pgd_freq=5,
    verbose=verbose,
    tol=tol,
    cluster_updates=cluster_updates,
)
_, primals_10, gaps_10, time = hybrid_cd(
    X,
    y,
    alphas,
    adaptive=False,
    pgd_freq=10,
    max_epochs=max_epochs,
    verbose=verbose,
    tol=tol,
    cluster_updates=cluster_updates,
)
beta_ada, primals_ada, gaps_ada, time_ada = hybrid_cd(
    X,
    y,
    alphas,
    adaptive=True,
    adaptive_tol = 1e-3,
    adaptive_patience = 2,
    max_epochs=max_epochs,
    verbose=verbose,
    tol=tol,
    cluster_updates=cluster_updates,
)

plt.clf()

plt.semilogy(gaps, label="cd_freq5")
# plt.semilogy(gaps_10, label="cd_freq10")
plt.semilogy(gaps_ada, label="cd_adaptive")

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(f"simulated data, n={n}, p={p}, rho={rho}, q={q}, reg={reg}")
plt.show(block=False)
