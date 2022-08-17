import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.data import get_data
from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

dataset = "real-sim"
# dataset = "Rhee2006"
# dataset = "bcTCGA"
# dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=10, n_features=10, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = get_data(dataset)
# X = X.todense()
fit_intercept = False

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.01

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, (y - fit_intercept * np.mean(y)) / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg

max_epochs = 10000
max_time = np.inf
tol = 1e-6

(
    beta_cd_reduced,
    intercept_cd_reduced,
    primals_cd_reduced,
    gaps_cd_reduced,
    time_cd_reduced,
) = hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    max_time=max_time,
    use_reduced_X=True,
    cluster_updates=True,
)

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    max_time=max_time,
    use_reduced_X=False,
    cluster_updates=True,
)

primals_star = np.min(np.hstack((primals_cd, np.array(primals_cd_reduced))))

plt.clf()
plt.semilogy(time_cd, primals_cd - primals_star, label="cd")
plt.semilogy(time_cd_reduced, primals_cd_reduced - primals_star, label="cd_reduced")

# plt.semilogy(time_cd, gaps_cd, label="cd")
# plt.semilogy(time_cd_reduced, gaps_cd_reduced, label="cd_updates")
# plt.xlabel("Time (s)")

# plt.semilogy(np.arange(len(gaps_cd))*10, gaps_cd, label="cd")
# plt.semilogy(np.arange(len(gaps_cd_reduced))*10, gaps_cd_reduced, label="cd_reduced")
# plt.xlabel("Epoch")

plt.ylabel("suboptimality")
plt.legend()
plt.title(dataset)
plt.show(block=False)
