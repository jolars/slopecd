import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from numpy.linalg import norm
from scipy import sparse, stats

from slope.solvers import hybrid_cd, oracle_cd, prox_grad
from slope.utils import dual_norm_slope

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(
        n_samples=100, n_features=10, random_state=0, rho=0.9
    )
else:
    X, y = fetch_libsvm(dataset)

n_samples, n_features = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.1

fit_intercept = True

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, (y - np.mean(y) * fit_intercept) / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 100

max_epochs = 10000
tol = 1e-6

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
)
beta_pgd, intercept_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    fista=False,
)
beta_oracle, intercept_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
)

# Duality gap vs epoch
plt.clf()

primal_star = np.min([np.min(primals_cd), np.min(primals_pgd), np.min(primals_oracle)])

plt.semilogy(primals_cd - primal_star, label="cd")
plt.semilogy(primals_pgd - primal_star, label="pgd")
plt.semilogy(primals_oracle - primal_star, label="oracle")

plt.ylabel("primal suboptimality")
plt.xlabel("epoch")
plt.legend()
plt.title(dataset)
plt.show(block=False)
