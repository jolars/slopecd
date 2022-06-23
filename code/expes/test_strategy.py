import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from scipy import stats

from slope.solvers import hybrid_cd, prox_grad
from slope.utils import dual_norm_slope
from slope.solvers import prox_grad
from slope.solvers import hybrid_cd
from slope.solvers import oracle_cd

dataset = "bcTCGA"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = get_data(dataset)

fit_intercept=True

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.01

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, (y - fit_intercept*np.mean(y)) / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg
plt.close("all")

max_epochs = 10000
max_time = 120
tol = 1e-6

beta_cd, primals_cd, gaps_cd, time_cd = hybrid_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, max_time=max_time
)
beta_pgd, primals_pgd, gaps_pgd, time_pgd = prox_grad(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, fista=True
)
beta_oracle, primals_oracle, gaps_oracle, time_oracle = oracle_cd(
    X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol,
)

plt.clf()

plt.semilogy(gaps_cd, label='cd')
plt.semilogy(gaps_pgd, label='pgd')
plt.semilogy(gaps_oracle, label='oracle')

plt.legend()
plt.title(dataset)
plt.show(block=False)

# Duality gap vs epoch for hybrid vs oracle
plt.clf()

plt.semilogy(
    np.arange(len(gaps_cd)), primals_cd - primals_pgd[-1], label='cd')
plt.semilogy(
    np.arange(len(gaps_oracle)), primals_oracle - primals_pgd[-1],
    label='oracle')

plt.legend()
plt.show(block=False)

# Time vs. duality gap
plt.clf()

plt.semilogy(time_cd, gaps_cd, label='cd')
plt.semilogy(time_pgd, gaps_pgd, label='pgd')
plt.semilogy(time_oracle, gaps_oracle, label='oracle')

plt.ylabel("duality gap")
plt.xlabel("Time (s)")
plt.legend()
plt.title(dataset)
plt.show(block=False)
