import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

dataset = "rcv1.binary"
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

max_epochs = 10000
tol = 1e-10

beta_cd, _, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=2, verbose=True, tol=tol
)
freqs = np.arange(1, 10)

betas = list()
primals = list()
gaps = list()
times = list()
for k in freqs:
    beta_cd, _, primals_cd, gap_cd, time_cd = hybrid_cd(
        X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, pgd_freq=k
    )
    betas.append(beta_cd)
    primals.append(primals_cd)
    gaps.append(gap_cd)
    times.append(time_cd)

# Time vs. duality gap
plt.clf()

for k in freqs:
    plt.semilogy(times[k - 1], gaps[k - 1], label="pgd freq = %s" % k)

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(dataset)
plt.show(block=False)
