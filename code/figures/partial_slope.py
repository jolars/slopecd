from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.random import default_rng

from figures import figspec
from slope.clusters import get_clusters
from slope.utils import lambda_sequence, sl1_norm


def signbit(x):
    if x >= 0:
        return 1
    else:
        return -1


def get_epsilon_c(c):
    m = len(c)
    epsilon_c = np.inf

    for i in range(m):
        epsilon_c = min(epsilon_c, np.abs(c[i]))
        for j in range(m):
            if i != j:
                epsilon_c = min(epsilon_c, np.abs(c[i] - c[j]))

    return epsilon_c * 0.9


def beta_update(beta, z, ind):
    beta_out = beta.copy()
    beta_out[ind] = z * np.sign(beta[ind])
    return beta_out


def get_cluster(z, beta, k):
    beta_new = beta_update(beta, z, ind)
    c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta_new)


def directional_derivative(z, delta, k, beta, lambdas):
    c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta)
    ind = c_ind[c_ptr[k] : c_ptr[k + 1]]
    c_k = np.delete(c[:n_c], k)
    epsilon_c = get_epsilon_c(c_k)

    if z == 0:
        upd = epsilon_c
    elif abs(z) in c_k:
        upd = z + epsilon_c * delta
    else:
        upd = z

    beta_clust = beta_update(beta, upd, ind)
    c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta_clust)
    new_pos = n_c - 1 - bisect_right(c_k[::-1], abs(upd))
    lambda_sum = np.sum(lambdas[c_ptr[new_pos] : c_ptr[new_pos + 1]])

    if z == 0:
        out = lambda_sum
    else:
        out = np.sign(z) * delta * lambda_sum

    beta_new = beta_update(beta, z, ind)
    x_tilde = X[:, ind] @ np.sign(beta[ind])
    r = X @ beta_new - y
    grad = x_tilde.T @ r / n

    return out + grad * delta


n = 10

beta = np.array([-3.0, 1.0, 3.0, 2.0])
p = len(beta)

rng = default_rng(1)

# cov = [[1, 0.85], [0.85, 1]]
sigma = np.zeros((p, p))
sigma[:] = 0.9
np.fill_diagonal(sigma, 1.0)
mu = np.ones(p)
X = rng.multivariate_normal(size=n, mean=mu, cov=sigma)
beta_true = beta + rng.standard_normal(p)

y = X @ beta_true

q = 0.6
reg = 0.8
fit_intercept = False
k = 0
delta = 1

c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta)
lambdas = lambda_sequence(X, y, fit_intercept, reg, q)

eps = 1e-6

c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta)
c_k = np.delete(c[:n_c], k)

ind = c_ind[c_ptr[k] : c_ptr[k + 1]]


plt.close("all")

plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots(
    1,
    1,
    figsize=(figspec.HALF_WIDTH, figspec.HALF_WIDTH * 0.6),
    constrained_layout=True,
)

x_min = -max(c) - 0.5
x_max = max(c) + 0.5
x_margin = (x_max - x_min) * plt.margins()[1]
x_lim = (x_min - x_margin, x_max + x_margin)

legend_symbols = [
    Line2D([0], [0], color=c, linestyle="-", marker="o", markerfacecolor=c)
    for c in ["tab:orange", "tab:blue"]
]
legend_labels = [r"$1$", r"$-1$"]

zs = np.sort(np.hstack((-c_k, [0.0], c_k, np.linspace(x_lim[0], x_lim[1], 100))))

obj = [sl1_norm(beta_update(beta, z, ind), lambdas) for z in zs]

y_lim = (np.min(obj), np.max(obj))
ps = np.hstack((-c_k, [0.0], c_k[::-1]))

ax.vlines(ps, *y_lim, color="darkgrey", linestyle="dotted")

ax.plot(zs, obj, color="black")
ax.set_ylabel(r"$H(z)$")

# ax.set_xlabel(r"$z$")

ax.set_xlim(*x_lim)
old_labels = ax.get_xticklabels()
ax.set_xticks(np.hstack([ax.get_xticks(), -c_k, c_k]))
ax.set_xticklabels(
    np.hstack(
        [
            old_labels,
            [f"$-c_{k}$" for k in range(2, 4)],
            [f"$c_{k}$" for k in range(2, 4)],
        ]
    )
)

plt.show(block=False)


savefig = True
if savefig:
    plt.savefig("../figures/partial_slope.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.savefig("../figures/partial_slope.svg", bbox_inches="tight", pad_inches=0.01)
