import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import admm, hybrid_cd, oracle_cd, prox_grad
from slope.utils import dual_norm_slope

n = 500
p = 100

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

n, p = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

reg = 0.01

alphas = alpha_max * alphas_seq * reg

max_epochs = 1000
tol = 1e-8

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, cluster_updates=True
)
beta_pgd, intercept_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, fista=True
)
beta_admm, _, primals_admm, gaps_admm, time_admm = admm(
    X,
    y,
    alphas,
    rho=1,
    adaptive_rho=True,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    fit_intercept=False,
)

beta_oracle, intercept_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol
)

# Duality gap vs epoch
# primal_star = np.min([np.min(primals_cd), np.min(primals_pgd), np.min(primals_admm)])
primal_star = np.min([np.min(primals_pgd), np.min(primals_admm)])

plt.clf()

plt.semilogy(time_cd, primals_cd - primal_star, label="cd")
plt.semilogy(time_pgd, primals_pgd - primal_star, label="pgd")
plt.semilogy(time_admm, primals_admm - primal_star, label="admm")
plt.semilogy(time_oracle, primals_oracle - primal_star, label="oracle")

plt.legend()
plt.title(f"{dataset}, n: {n}, p: {p} reg: {reg}, q: {q}")
plt.show(block=False)

plt.legend()
plt.show(block=False)
