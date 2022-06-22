import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from matplotlib.cm import get_cmap
from scipy import sparse, stats

from slope.data import get_data
from slope.solvers import hybrid_cd, newt_alm, prox_grad
from slope.utils import dual_norm_slope

cm = get_cmap("tab10")

n = 100
p = 10000
rho = 0.3

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=rho)
else:
    X, y = get_data(dataset)

n, p = X.shape

randnorm = stats.norm(loc=0, scale=1)
q = 0.2
reg = 0.2

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq * reg

max_epochs = 10_000
gap_freq = 10
tol = 1e-6
verbose = True

w_ista, primals_ista, gaps_ista, times_ista = prox_grad(
    X,
    y,
    lambdas,
    fista=True,
    gap_freq=gap_freq,
    max_epochs=max_epochs,
    tol=tol,
    verbose=verbose,
)

w_cd, primals_cd, gaps_cd, times_cd = hybrid_cd(
    X, y, lambdas, max_epochs=max_epochs, tol=tol, verbose=verbose, cluster_updates=True
)

w_newt, primals_newt, gaps_newt, times_newt = newt_alm(
    X,
    y,
    lambdas,
    gap_freq=1,
    max_epochs=max_epochs,
    tol=tol,
    verbose=verbose,
)

plt.close("all")
plt.figure()

title = f"{dataset}, q: {q}, reg: {reg}"
if dataset == "simulated":
    title += f", rho: {rho}"

plt.title(title)

primals_ista = np.array(primals_ista)
primals_newt = np.array(primals_newt)
primals_cd = np.array(primals_cd)
p_star = min(np.min(primals_ista), np.min(primals_newt), np.min(primals_cd))
plt.semilogy(times_ista, primals_ista - p_star, c=cm(0), label="primal subopt PGD")
plt.semilogy(times_newt, primals_newt - p_star, c=cm(1), label="primal subopt NEWT-ALM")
plt.semilogy(times_cd, primals_cd - p_star, c=cm(2), label="primal subopt cd")

plt.legend()

plt.show(block=False)
