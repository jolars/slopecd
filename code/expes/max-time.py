import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd, oracle_cd, prox_grad
from slope.utils import dual_norm_slope

dataset = "real-sim"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5
plt.close("all")

max_epochs = 10000
tol = 1e-10
max_time = 3

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, max_time=max_time
)
beta_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    fista=True,
    max_time=max_time,
)
beta_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, max_time=max_time
)

plt.clf()

primal_star = np.min([np.min(primals_oracle), np.min(primals_pgd), np.min(primals_cd)])

plt.semilogy(time_cd, primals_cd - primal_star, label="cd")
plt.semilogy(time_pgd, primals_pgd - primal_star, label="pgd")
plt.semilogy(time_oracle, primals_oracle - primal_star, label="oracle")

plt.ylabel("suboptimality")
plt.xlabel("time (s)")
plt.legend()
plt.title(dataset)
plt.show(block=False)
