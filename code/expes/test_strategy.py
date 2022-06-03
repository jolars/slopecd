import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.solvers import hybrid_cd, prox_grad
from slope.utils import dual_norm_slope
from slope.data import get_data

dataset = "bcTCGA"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = get_data(dataset)

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.01

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg
plt.close("all")

max_epochs = 10000
max_time = 120
tol = 1e-6

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, line_search=False,max_time=max_time
)
beta_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, max_time=max_time,
    fista=True
)

plt.clf()

plt.semilogy(time_cd, gaps_cd, label="cd")
plt.semilogy(time_pgd, gaps_pgd, label="fista")

plt.legend()
plt.title(f"{dataset}, reg: {reg}, q: {q}")
plt.show(block=False)
