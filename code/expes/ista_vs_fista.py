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
p = 500

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0)
else:
    X, y = fetch_libsvm(dataset)

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5
alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq / 50

plt.close('all')
plt.figure()

max_epochs = 20_000

w_ista, E_ista, gaps_ista, _ = prox_grad(
    X, y, lambdas, fista=False, verbose=True, gap_freq=10,
    max_epochs=max_epochs)

w_fista, E_fista, gaps_fista, _ = prox_grad(
    X, y, lambdas, fista=True, verbose=True, gap_freq=10,
    max_epochs=max_epochs)


E_ista = np.array(E_ista)
E_fista = np.array(E_fista)
p_star = min(np.min(E_ista), np.min(E_fista))
plt.figure()
plt.semilogy(E_ista - p_star, c=cm(0), label='primal subopt PGD')
plt.semilogy(E_fista - p_star, c=cm(1), label='primal subopt APGD')

plt.semilogy(gaps_ista, c=cm(0), linestyle='--', label='gap PGD')
plt.semilogy(gaps_fista, c=cm(1), linestyle='--', label='gap APGD')
plt.legend()
plt.show(block=False)
