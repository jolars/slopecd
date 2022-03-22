import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from scipy import stats
from libsvmdata import fetch_libsvm
from benchopt.datasets.simulated import make_correlated_data

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope
cm = get_cmap('tab10')

n = 100
p = 2000
rho = 0.5

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

lambdas = alpha_max * alphas_seq / 50


max_epochs = 20_000
tol = 1e-7

_, E_ista, gaps_ista, times_ista = prox_grad(
    X, y, lambdas, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

_, E_ista_lsbb, gaps_ista_lsbb, times_ista_lsbb = prox_grad(
    X, y, lambdas, line_search=True, barzilai=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs, tol=tol)

_, E_ista_ls, gaps_ista_ls, times_ista_ls = prox_grad(
    X, y, lambdas, line_search=True, verbose=True, gap_freq=10,
    ls_type=0,
    max_epochs=max_epochs, tol=tol)

_, E_ista_ls1, gaps_ista_ls1, times_ista_ls1 = prox_grad(
    X, y, lambdas, line_search=True, verbose=True, gap_freq=10,
    ls_type=1,
    max_epochs=max_epochs, tol=tol)

_, E_ista_ls2, gaps_ista_ls2, times_ista_ls2 = prox_grad(
    X, y, lambdas, line_search=True, verbose=True, gap_freq=10,
    ls_type=2,
    max_epochs=max_epochs, tol=tol)

_, E_ista_ls3, gaps_ista_ls3, times_ista_ls3 = prox_grad(
    X, y, lambdas, line_search=True, barzilai=True, verbose=True, gap_freq=10,
    ls_type=3,
    max_epochs=max_epochs, tol=tol)

# _, E_anderson, gaps_anderson, times_anderson = prox_grad(
#     X, y, lambdas, fista=False, anderson=True, verbose=True, gap_freq=10,
#     max_epochs=max_epochs, tol=tol)

# _, E_fista_ls, gaps_fista_ls, _ = prox_grad(
#     X, y, lambdas, fista=True, verbose=True, gap_freq=10,
#     max_epochs=max_epochs, line_search=True, tol=tol)

# _, E_fista, gaps_fista, times_fista = prox_grad(
#     X, y, lambdas, fista=True, verbose=True, gap_freq=10,
#     max_epochs=max_epochs, tol=tol)


# plt.close('all')
# plt.figure()
plt.clf()
# p_star = min(np.min(E_ista), np.min(E_fista), np.min(E_fista_ls))
p_star = min(np.min(E_ista), np.min(E_ista_ls), np.min(E_ista_lsbb))
plt.semilogy(times_ista, E_ista - p_star, c=cm(0), label='primal subopt PGD')
# plt.semilogy(E_fista - p_star, c=cm(1), label='primal subopt APGD')
# plt.semilogy(E_fista_ls - p_star, c=cm(2), label='primal subopt APGD+LS')
# plt.semilogy(E_anderson - p_star, c=cm(3), label='primal subopt PGD Anderson')
plt.semilogy(times_ista_ls, E_ista_ls - p_star, c=cm(1), label='primal subopt PGD+LS')
plt.semilogy(times_ista_ls1, E_ista_ls1 - p_star, c=cm(2), label='primal subopt PGD+LS1')
plt.semilogy(times_ista_ls2, E_ista_ls2 - p_star, c=cm(3), label='primal subopt PGD+LS2')
plt.semilogy(times_ista_ls3, E_ista_ls3 - p_star, c=cm(4), label='primal subopt PGD+LS3')
plt.semilogy(times_ista_lsbb, E_ista_lsbb - p_star, c=cm(5), label='primal subopt PGD+LSBB')

# plt.semilogy(times_ista, gaps_ista, c=cm(0), linestyle='--', label='gap PGD')
# plt.semilogy(gaps_fista, c=cm(1), linestyle='--', label='gap APGD')
# plt.semilogy(gaps_fista_ls, c=cm(2), linestyle='--', label='gap APGD+LS')
# plt.semilogy(gaps_anderson, c=cm(3), linestyle='--', label='gap PGD Anderson')
# plt.semilogy(times_ista_ls, gaps_ista_ls, c=cm(4), linestyle='--', label='gap PGD+LS')
# plt.semilogy(times_ista_lsbb, gaps_ista_lsbb, c=cm(5), linestyle='--', label='gap PGD+LSBB')
plt.legend()
plt.xlabel("Time (s)")
plt.show(block=False)
