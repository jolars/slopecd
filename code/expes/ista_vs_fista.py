import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from scipy import stats
from libsvmdata import fetch_libsvm
from benchopt.datasets.simulated import make_correlated_data

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope
cm = get_cmap('tab10')

n = 200
p = 600
rho = 0.2

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=rho)
else:
    X, y = fetch_libsvm(dataset)

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5
alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq / 20

plt.close('all')
plt.figure()

max_epochs = 20_000
tol = 1e-8

_, E_ista, gaps_ista, times_ista = prox_grad(
    X, y, lambdas, fista=False, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

_, E_ista_ls, gaps_ista_ls, times_ista_ls = prox_grad(
    X, y, lambdas, fista=False, line_search=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

_, E_anderson, gaps_anderson, times_anderson = prox_grad(
    X, y, lambdas, fista=False, anderson=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

_, E_fista_ls, gaps_fista_ls, _ = prox_grad(
    X, y, lambdas, fista=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs, line_search=True, tol=tol)

_, E_fista, gaps_fista, times_fista = prox_grad(
    X, y, lambdas, fista=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

E_ista = np.array(E_ista)
E_fista = np.array(E_fista)
E_fista_ls = np.array(E_fista_ls)
p_star = min(np.min(E_ista), np.min(E_fista), np.min(E_fista_ls))
p_star = min(np.min(E_fista), np.min(E_fista_ls))
plt.semilogy(E_ista - p_star, c=cm(0), label='primal subopt PGD')
plt.semilogy(E_fista - p_star, c=cm(1), label='primal subopt APGD')
plt.semilogy(E_fista_ls - p_star, c=cm(2), label='primal subopt APGD+LS')
plt.semilogy(E_anderson - p_star, c=cm(3), label='primal subopt PGD Anderson')
plt.semilogy(E_ista_ls - p_star, c=cm(4), label='primal subopt PGD+LS')

plt.semilogy(gaps_ista, c=cm(0), linestyle='--', label='gap PGD')
plt.semilogy(gaps_fista, c=cm(1), linestyle='--', label='gap APGD')
plt.semilogy(gaps_fista_ls, c=cm(2), linestyle='--', label='gap APGD+LS')
plt.semilogy(gaps_anderson, c=cm(3), linestyle='--', label='gap PGD Anderson')
plt.semilogy(gaps_ista_ls, c=cm(4), linestyle='--', label='gap PGD+LS')
plt.legend()
plt.show(block=False)
