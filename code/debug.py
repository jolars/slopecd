import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.data import get_data
from slope.solvers import hybrid_cd, oracle_cd
from slope.utils import dual_norm_slope

X, y = get_data("Scheetz2006")
fit_intercept = False

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.01

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))
alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)
alphas = alpha_max * alphas_seq * reg

max_epochs = 100_000
max_time = np.inf
tol = 1e-10

beta_cd, intercept_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=max_epochs,
    verbose=True,
    tol=tol,
    max_time=max_time,
    cluster_updates=True,
)

beta_oracle, intercept_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
    X,
    y,
    alphas,
    fit_intercept=fit_intercept,
    max_epochs=50,
    verbose=True,
    tol=0,
    max_time=max_time,
    w_star=beta_cd
)

primals_star = np.min(np.hstack((np.array(primals_cd), np.array(primals_oracle))))

plt.close('all')

fig, axarr = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax = axarr[0]
ax.semilogy(time_cd, primals_cd - primals_star, label="cd")
ax.semilogy(time_oracle, primals_oracle - primals_star, label="cd_oracle")


ax.set_ylabel("suboptimality")
ax.legend()
ax.set_title(dataset)

ax = axarr[1]
ax.semilogy(times_cd, gaps_cd, label='cd')
ax.semilogy(times_oracle, gaps_oracle, label='cd_oracle')
ax.legend()
ax.set_ylabel("duality gap")
ax.set_xlabel("Time (s)")
plt.show(block=False)
