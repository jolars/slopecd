import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data

from slope.utils import dual_norm_slope
from slope.solvers import prox_grad, hybrid


X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5
plt.close('all')

for n_cd in [0, 1, 5, 10]:
    w, E, gaps, theta = prox_grad(
        X, y, alphas, max_iter=1000 // (n_cd + 1), n_cd=n_cd, verbose=1)
    print(gaps[0])

    plt.semilogy(np.arange(len(E)) * (1 + n_cd), gaps,
                 label=f'n_cd = {n_cd}')
w, E, gaps, theta = hybrid(
    X, y, alphas, max_iter=1000, verbose=1)
plt.semilogy(np.arange(len(E)), gaps,
             label='Hybrid strategy')
plt.legend()
plt.show(block=False)
