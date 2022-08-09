import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats
from scipy.sparse import csc_matrix

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

dataset = "news20.binary"
reg = 0.2

X, y = fetch_libsvm(dataset)
n, p = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.1

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg
max_epochs = 50
tol = 1e-8
verbose = True

plt.clf()

gaps_cd = []
gaps_cd_updates = []

beta_cd, _, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=verbose, tol=tol
)
beta_cd_updates, _, primals_cd_updates, gaps_cd_updates, time_cd_updates = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=verbose,
    tol=tol,
    cluster_updates=True,
)

plt.clf()

plt.semilogy(time_cd, gaps_cd, label="cd")
plt.semilogy(time_cd_updates, gaps_cd_updates, label="cd_updates")

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(f"dataset={dataset}, n={n}, p={p}, reg={reg}")
plt.show(block=False)
