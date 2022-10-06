import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
from scipy import stats

from slope.data import get_data
from slope.solvers import admm, hybrid_cd, newt_alm, oracle_cd, prox_grad
from slope.utils import lambda_sequence

dataset = "bcTCGA"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
else:
    X, y = get_data(dataset)

n, p = X.shape
fit_intercept = True
q = 0.1

lambdas = lambda_sequence(X, y, fit_intercept, 1, q)

# fit_interecpt = True
# max_epochs = 10000
# max_time = 100
# q = 0.1
# reg = 0.02
# verbose = True
# tol = 1e-4

# # beta_pgd, intercept_cd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
# #     X,
# #     y,
# #     lambdas,
# #     fit_intercept=fit_intercept,
# #     max_epochs=max_epochs,
# #     verbose=verbose,
# #     tol=tol,
# #     fista=True,
# # )

# beta_admm, intercept_cd, primals_admm, gaps_admm, time_admm = admm(
#     X,
#     y,
#     lambdas,
#     fit_intercept=fit_intercept,
#     max_epochs=max_epochs,
#     verbose=verbose,
#     tol=tol,
# )

# beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd, _ = hybrid_cd(
#     X,
#     y,
#     lambdas,
#     fit_intercept=fit_intercept,
#     max_epochs=max_epochs,
#     verbose=verbose,
#     tol=tol,
#     max_time=max_time,
# )

# # beta_oracle, intercept_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
# #     X,
# #     y,
# #     lambdas,
# #     fit_intercept=fit_intercept,
# #     max_epochs=max_epochs,
# #     verbose=verbose,
# #     tol=tol,
# # )

# # beta_newt, intercept_newt, primals_newt, gaps_newt, time_newt = newt_alm(
# #     X,
# #     y,
# #     lambdas,
# #     fit_intercept=fit_intercept,
# #     max_epochs=max_epochs,
# #     verbose=verbose,
# #     tol=tol,
# # )

# null_dev = 0.5 * norm(y - np.mean(y) * fit_intercept) ** 2
# dev = 0.5 * norm(y - X @ beta_cd - intercept_cd) ** 2
# dev_ratio = 1 - dev / null_dev

# nnz = np.sum(np.unique(np.abs(beta_cd)) != 0)

# plt.close("all")

# plt.title(dataset)

# plt.semilogy(time_cd, gaps_cd, label="cd")
# plt.semilogy(time_admm, gaps_admm, label="admm")
# # plt.semilogy(time_pgd, gaps_pgd, label="pgd")
# # plt.semilogy(time_oracle, gaps_oracle, label="oracle")

# plt.ylabel("duality gap")
# plt.xlabel("Time (s)")

# plt.legend()

# plt.show(block=False)
