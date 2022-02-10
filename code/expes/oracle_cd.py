import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data

from slope.utils import dual_norm_slope
from slope.solvers import prox_grad, oracle_cd


X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5
plt.close('all')

w, E, gaps, theta = prox_grad(
    X, y, alphas, max_iter=1000, n_cd=0, verbose=1)

w_oracle, E_oracle = oracle_cd(X, y, alphas, 1000, tol=0)


E_min = min(np.min(E), np.min(E_oracle))

plt.semilogy(E - E_min, label="PGD")
plt.semilogy(E_oracle - E_min, label="oracle")
plt.legend()
plt.show(block=False)
