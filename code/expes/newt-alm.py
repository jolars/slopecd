from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from libsvmdata import fetch_libsvm
from matplotlib.cm import get_cmap
from scipy import stats

import slope

# from slope.solvers import newt_alm, prox_grad
# from slope.utils import dual_norm_slope

x = np.array([5, 2.0, 1.0])
lambdas = np.array([2.5, 1.0, 1.0])
epsilon = 10

slope.sorted_l1_proj(x, lambdas, epsilon)
# z = project_to_OWL_ball(z - opts.gamma *(AtA*z - Atb), w, epsilon, 'false');

# cm = get_cmap("tab10")

# n = 1000
# p = 10

# dataset = "simulated"
# if dataset == "simulated":
#     X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0)
# else:
#     X, y = fetch_libsvm(dataset)

# y = y - np.mean(y)

# randnorm = stats.norm(loc=0, scale=1)
# q = 0.1
# alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


# alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

# lambdas = alpha_max * alphas_seq / 100

# plt.close("all")
# plt.figure()

# max_epochs = 100
# gap_freq = 10

# w_ista, primals_ista, gaps_ista, times_ista = prox_grad(
#     X, y, lambdas, fista=False, verbose=True, gap_freq=gap_freq, max_epochs=max_epochs
# )

# w_newt, primals_newt, gaps_newt, times_newt = newt_alm(
#     X, y, lambdas, verbose=True, gap_freq=gap_freq, max_epochs=max_epochs
# )

# primals_ista = np.array(primals_ista)
# primals_fista = np.array(primals_newt)
# p_star = min(np.min(primals_ista), np.min(primals_fista))
# plt.semilogy(primals_ista - p_star, c=cm(0), label="primal subopt PGD")
# plt.semilogy(primals_fista - p_star, c=cm(1), label="primal subopt NEWT-ALM")

# plt.semilogy(gaps_ista, c=cm(0), linestyle="--", label="gap PGD")
# plt.semilogy(gaps_newt, c=cm(1), linestyle="--", label="gap NEWT-ALM")
# plt.legend()
# plt.show(block=False)
