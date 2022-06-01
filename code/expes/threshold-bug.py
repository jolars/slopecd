import time

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd, oracle_cd, prox_grad
from slope.utils import dual_norm_slope

# from slope.solvers.proxsplit_cd import proxsplit_cd

random_state=5

X, y, _ = make_correlated_data(n_samples=100, n_features=10, rho=0.9, random_state=random_state)

randnorm = stats.norm(loc=0, scale=1)
q = 0.4

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 5

max_epochs = 50
tol = 1e-4

# y = y - np.mean(y)
# X = X - np.mean(X,axis=0)


beta_cd_old, primals_cd_old, gaps_cd_old, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    cluster_updates=True,
    use_old_thresholder=True
)

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    cluster_updates=True,
    use_old_thresholder=False
)

print(f"decreasing: {np.all(np.diff(primals_cd) < 0)}")

plt.clf()

plt.semilogy(primals_cd, label="cd_primal_new")
plt.semilogy(primals_cd_old, label="cd_primal")

plt.legend()
plt.show(block=False)
