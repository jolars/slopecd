import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

rho = 0.9
n = 100
p = 5000
reg = 0.5

X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=rho)

n, p = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg
max_epochs = 500
tol = 1e-10
n_it = 20
verbose = False

times_cd = np.empty((n_it, max_epochs))
times_cd_updates = np.empty((n_it, max_epochs))

min_epoch = max_epochs

plt.clf()

gaps_cd = []
gaps_cd_updates = []

for i in range(n_it):
    beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
        X, y, alphas, max_epochs=max_epochs, verbose=verbose, tol=tol
    )
    beta_cd_updates, primals_cd_updates, gaps_cd_updates, time_cd_updates = hybrid_cd(
        X,
        y,
        alphas,
        max_epochs=max_epochs,
        verbose=verbose,
        tol=tol,
        cluster_updates=True,
    )
    times_cd[i, : len(time_cd)] = time_cd
    times_cd_updates[i, : len(time_cd_updates)] = time_cd_updates
    min_epoch = min(min_epoch, len(time_cd))
    min_epoch = min(min_epoch, len(time_cd_updates))

times_cd = times_cd[:, :min_epoch]
times_cd_updates = times_cd_updates[:, :min_epoch]
time_cd = np.mean(times_cd, axis=0)
time_cd_updates = np.mean(times_cd_updates, axis=0)

gaps_cd = gaps_cd[:min_epoch]
gaps_cd_updates = gaps_cd_updates[:min_epoch]

plt.clf()

plt.semilogy(time_cd, gaps_cd, label="cd")
plt.semilogy(time_cd_updates, gaps_cd_updates, label="cd_updates")
lo, up = stats.bootstrap((times_cd,), np.mean, axis=-2).confidence_interval
plt.fill_betweenx(gaps_cd, lo, up, alpha=0.2)
lo, up = stats.bootstrap((times_cd_updates,), np.mean, axis=-2).confidence_interval
plt.fill_betweenx(gaps_cd_updates, lo, up, alpha=0.2)

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(f"simulated data, n={n}, p={p}, rho={rho}")
plt.show(block=False)
