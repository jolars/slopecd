import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.data import get_data
from slope.solvers import admm, hybrid_cd, prox_grad
from slope.utils import dual_norm_slope, preprocess

dataset = "bcTCGA"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = get_data(dataset)

X, y = preprocess(X, y)

fit_intercept = True

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.1

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, (y - fit_intercept * np.mean(y)) / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg

max_epochs = 10000
max_time = 120
tol = 1e-8

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    max_time=max_time,
)

beta_pgd, intercept_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    fista=True,
)

beta_admm, intercept_admm, primals_admm, gaps_admm, time_admm = admm(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
)

# Time vs. duality gap
plt.clf()

plt.semilogy(time_cd, gaps_cd, label="cd")
plt.semilogy(time_pgd, gaps_pgd, label="pgd")
plt.semilogy(time_admm, gaps_admm, label="admm")

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(dataset)
plt.show(block=False)
