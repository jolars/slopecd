import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from libsvmdata import fetch_libsvm
from matplotlib.cm import get_cmap
from scipy import stats

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope

cm = get_cmap("tab10")

n = 100
p = 2000
rho = 0.5

reg = 0.01
q = 0.1

dataset = "rcv1.binary"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=rho)
else:
    X, y = fetch_libsvm(dataset)

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq * reg


max_epochs = 200
tol = 1e-4
gap_freq = 10

_, E_ista, gaps_ista, times_ista = prox_grad(
    X, y, lambdas, verbose=True, gap_freq=10, max_epochs=max_epochs, tol=tol
)

_, E_ista_lsbb, gaps_ista_lsbb, times_ista_lsbb = prox_grad(
    X,
    y,
    lambdas,
    bb_step_size=True,
    line_search=True,
    verbose=True,
    gap_freq=10,
    max_epochs=max_epochs,
    tol=tol,
)

_, E_ista_ls, gaps_ista_ls, times_ista_ls = prox_grad(
    X,
    y,
    lambdas,
    line_search=True,
    verbose=True,
    gap_freq=gap_freq,
    max_epochs=max_epochs,
    tol=tol,
)

_, E_anderson, gaps_anderson, times_anderson = prox_grad(
    X,
    y,
    lambdas,
    acceleration="anderson",
    verbose=True,
    gap_freq=gap_freq,
    max_epochs=max_epochs,
    tol=tol,
)

_, E_fista, gaps_fista, times_fista = prox_grad(
    X,
    y,
    lambdas,
    acceleration="fista",
    verbose=True,
    gap_freq=gap_freq,
    max_epochs=max_epochs,
    tol=tol,
)

_, E_fista_ls, gaps_fista_ls, times_fista_ls = prox_grad(
    X,
    y,
    lambdas,
    acceleration="fista",
    verbose=True,
    gap_freq=gap_freq,
    max_epochs=max_epochs,
    line_search=True,
    tol=tol,
)

plt.clf()

p_star = min(np.min(E_ista), np.min(E_fista), np.min(E_fista_ls), np.min(E_ista_lsbb))
plt.semilogy(times_ista, E_ista - p_star, c=cm(0), label="primal subopt PGD")
plt.semilogy(times_ista_ls, E_ista_ls - p_star, c=cm(1), label="primal subopt PGD+LS")
plt.semilogy(
    times_ista_lsbb, E_ista_lsbb - p_star, c=cm(5), label="primal subopt PGD+LS+BB"
)
plt.semilogy(
    times_anderson, E_anderson - p_star, c=cm(6), label="primal subopt PGD Anderson"
)
plt.semilogy(times_fista, E_fista - p_star, c=cm(7), label="primal subopt FISTA")
plt.semilogy(times_fista_ls, E_fista_ls - p_star, c=cm(8), label="primal subopt FISTA+LS")

# plt.semilogy(times_ista, gaps_ista, c=cm(0), linestyle='--', label='gap PGD')
# plt.semilogy(gaps_fista, c=cm(1), linestyle='--', label='gap APGD')
# plt.semilogy(gaps_fista_ls, c=cm(2), linestyle='--', label='gap APGD+LS')
# plt.semilogy(gaps_anderson, c=cm(3), linestyle='--', label='gap PGD Anderson')
# plt.semilogy(times_ista_ls, gaps_ista_ls, c=cm(4), linestyle='--', label='gap PGD+LS')
# plt.semilogy(times_ista_lsbb, gaps_ista_lsbb, c=cm(5), linestyle='--', label='gap PGD+LSBB')

plt.legend()
plt.title(f"{dataset}, reg = {reg}, q = {q}")

plt.xlabel("Time (s)")
plt.show(block=False)
