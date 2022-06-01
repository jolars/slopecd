import numpy as np
from scipy import stats
from scipy import sparse
import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data
from slope.utils import dual_norm_slope
from slope.solvers import prox_grad
from slope.solvers import hybrid_cd
# from slope.solvers import oracle_cd

from libsvmdata import fetch_libsvm

dataset = "real-sim"
if dataset == "simulated":
    X, y, _ = make_correlated_data(
        n_samples=100, n_features=10, random_state=0, rho=0.9)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

# X = X.toarray()
# X1 = np.hstack((np.ones((X.shape[0], 1)), X))

# np.linalg.norm(X, 2)
# np.linalg.norm(X1, 2)

randnorm = stats.norm(loc=0, scale=1)
q = 0.1

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5

max_epochs = 10000
tol = 1e-6

fit_intercept=True

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, fit_intercept=fit_intercept, max_epochs=max_epochs, verbose=True, tol=tol)
beta_pgd, intercept_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X, y, alphas, fit_intercept=fit_intercept, max_epochs=max_epochs, verbose=True, tol=tol, fista=False
)
# beta_oracle, intercept_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
#     X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol,
# )

# Duality gap vs epoch
plt.clf()

plt.semilogy(primals_cd, label='cd')
plt.semilogy(primals_pgd, label='pgd')
# plt.semilogy(gaps_cd, label='cd')
# plt.semilogy(gaps_pgd, label='pgd')
# plt.semilogy(np.arange(len(gaps_oracle)), gaps_oracle, label='oracle')

plt.legend()
plt.title(dataset)
plt.show(block=False)

