import time

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd, oracle_cd, prox_grad
from slope.utils import dual_norm_slope

# from slope.solvers.proxsplit_cd import proxsplit_cd

random_state=6

X, y, _ = make_correlated_data(n_samples=100, n_features=10, rho=0.9, random_state=random_state)

randnorm = stats.norm(loc=0, scale=1)
q = 0.4

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5

max_epochs = 10
tol = 1e-4

# y = y - np.mean(y)
# X = X - np.mean(X,axis=0)

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    cluster_updates=True,
)


beta_cd_old, primals_cd_old, gaps_cd_old, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    cluster_updates=True,
    use_old_thresholder=True,
)

# # beta_zero, primals_zero, gaps_zero, time_zero = hybrid_cd(
#     X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, cluster_updates=True,update_zero_cluster=True)
beta_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    fista=False,
)
# # beta_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
# #     X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol,
# # )

# # Duality gap vs epoch
plt.clf()

# # plt.semilogy(np.arange(len(gaps_cd)), gaps_cd, label='cd')
# plt.semilogy(gaps_pgd, label='pgd_gap')
# plt.semilogy(gaps_cd, label='cd_gap')

plt.semilogy(primals_pgd, label="pgd_primal")
plt.semilogy(primals_cd, label="cd_primal")
plt.semilogy(primals_cd_old, label="cd_primal_old")

# plt.semilogy(time_zero, gaps_zero, label='cd_zero')

print(f"decreasing: {np.all(np.diff(primals_cd) < 0)}")

plt.legend()
plt.show(block=False)
