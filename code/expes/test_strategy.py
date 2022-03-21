import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data
from slope.utils import dual_norm_slope
from slope.solvers import prox_grad
from slope.solvers import hybrid_cd
from slope.solvers import oracle_cd
from slope.solvers.proxsplit_cd import proxsplit_cd

import time
from libsvmdata import fetch_libsvm

dataset = 'real-sim'
if dataset == "simulated":
    X, y, _ = make_correlated_data(
        n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5
plt.close('all')

max_epochs = 10000
tol = 1e-10


start_cd = time.time()
beta_cd, primals_cd, gaps_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol)
time_cd = time.time() - start_cd
start_pgd = time.time()
beta_star, primals_star, gaps_star, theta_star = prox_grad(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol,
)
time_pgd = time.time() - start_pgd

beta_oracle, primals_oracle, gaps_oracle = oracle_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol,
)
beta_cdsplit, primals_cdsplit, gaps_cdsplit, theta_cdsplit = proxsplit_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, split_freq=1
)


plt.semilogy(np.arange(len(gaps_cd)), gaps_cd, label='cd')
plt.semilogy(np.arange(len(gaps_star)), gaps_star, label='pgd')
plt.semilogy(np.arange(len(gaps_oracle)), gaps_oracle, label='oracle')
plt.semilogy(np.arange(len(gaps_cdsplit)), gaps_cdsplit, label='cd_proxsplit')

plt.legend()
plt.title(dataset)
plt.show()


plt.semilogy(
    np.arange(len(gaps_cd)), primals_cd - primals_star[-1], label='cd')
plt.semilogy(
    np.arange(len(gaps_oracle)), primals_oracle - primals_star[-1],
    label='oracle')

plt.legend()
plt.show()
