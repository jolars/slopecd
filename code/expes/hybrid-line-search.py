import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

dataset = "rcv1.binary"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.1

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg

max_epochs = 10000
max_time = 360
tol = 1e-6
beta_cd_ls, primals_cd_ls, gaps_cd_ls, time_cd_ls = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    line_search=True,
    max_time=max_time,
)

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    line_search=False,
    max_time=max_time,
)

plt.clf()

plt.semilogy(time_cd, gaps_cd, label="cd")
plt.semilogy(time_cd_ls, gaps_cd_ls, label="cd_ls")

plt.legend()
plt.title(f"{dataset}, reg: {reg}, q: {q}")
plt.show(block=False)
