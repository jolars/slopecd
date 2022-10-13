from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.random import default_rng

from figures import figspec
from slope.clusters import get_clusters
from slope.utils import lambda_sequence, primal


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
reg = 0.9
fit_intercept = False
k = 0
delta = 1

c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta)

lambdas = lambda_sequence(X, y, fit_intercept, reg, q)

# zs = np.sort(np.hstack(([0.0], c[:n_c], np.linspace(-6, 6, 100))))

eps = 1e-6

c, c_ptr, c_ind, c_perm, n_c = get_clusters(beta)
c_k = np.delete(c[:n_c], k)

ind = c_ind[c_ptr[k] : c_ptr[k + 1]]

C_PAD = 0.25


def plot_dirder(delta, x0, x1, ax):
    eps = 1e-6

    if delta > 0:
        starts = x0
        ends = x1 - eps
    else:
        starts = x0 + eps
        ends = x1

    dir_der_starts = [
        directional_derivative(z_i, delta, k, beta, lambdas) for z_i in starts
    ]
    dir_der_ends = [
        directional_derivative(z_i, delta, k, beta, lambdas) for z_i in ends
    ]

    if delta == 1:
        linecolor = "tab:orange"
    else:
        linecolor = "tab:blue"

    for i in range(len(ends)):
        x = (starts[i], ends[i])
        y = (dir_der_starts[i], dir_der_ends[i])
        ax.plot(x, y, color=linecolor, linestyle="-")

    ax.plot(
        starts[1:],
        dir_der_starts[1:],
        marker="o",
        linestyle="",
        markersize=4,
        color=linecolor,
        markerfacecolor=linecolor if delta == 1 else "white",
    )
    ax.plot(
        ends[:-1],
        dir_der_ends[:-1],
        marker="o",
        linestyle="",
        markersize=4,
        color=linecolor,
        markerfacecolor="white" if delta == 1 else linecolor,
    )


plt.close("all")

plt.rcParams["text.usetex"] = True

fig, axs = plt.subplots(
    2,
    1,
    figsize=(figspec.HALF_WIDTH, figspec.HALF_WIDTH * 1.25),
    constrained_layout=True,
    sharex=True,
    gridspec_kw=dict(height_ratios=(0.25, 0.75)),
)

x_min = -max(c_k)
x_max = max(c_k)
x_margin = (x_max - x_min) * plt.margins()[1]
x_lim = (x_min - x_margin, x_max + x_margin)

ps = np.hstack((-c_k, [0.0], c_k[::-1]))

tmp = [-max(c), max(c)]
y_rng = np.array([directional_derivative(tmp_i, -1, k, beta, lambdas) for tmp_i in tmp])
y_rng2 = np.hstack((y_rng, -y_rng))
y_lim = (min(y_rng2), max(y_rng2))

axs[1].vlines(ps, y_lim[0], y_lim[1], color="darkgrey", linestyle="dotted")
axs[1].hlines(0.0, x_lim[0], x_lim[1], color="grey", linestyle="dashed")

x0 = np.hstack((-np.max(c_k) - C_PAD, -c_k, [0.0], c_k[::-1]))
x1 = np.hstack((-c_k, [0.0], c_k[::-1], np.max(c_k) + C_PAD))

plot_dirder(1, x0, x1, axs[1])
plot_dirder(-1, x0, x1, axs[1])

legend_symbols = [
    Line2D(
        [0], [0], color=c, linestyle="-", marker="o", markerfacecolor=c, markersize=4
    )
    for c in ["tab:orange", "tab:blue"]
]
legend_labels = [r"$1$", r"$-1$"]

axs[1].set_ylabel(r"$G'(z)$")
axs[1].set_xlabel(r"$z$")
axs[1].legend(legend_symbols, legend_labels, title=r"$\delta$")

zs = np.sort(np.hstack((-c_k, [0.0], c_k, np.linspace(x_lim[0], x_lim[1], 100))))

obj = [primal(beta_update(beta, z, ind), X, y, lambdas) for z in zs]


axs[0].vlines(ps, np.min(obj), np.max(obj), color="darkgrey", linestyle="dotted")
axs[0].plot(zs, obj, color="black")
axs[0].set_ylabel(r"$G(z)$")
# axs[0].set_xlabel(r"$z$")


plt.show(block=False)

savefig = True

if savefig:
    figpath = figspec.fig_path("directional-derivative")
    formats = [".svg", ".pdf"]

    plt.rcParams["text.usetex"] = True
    [
        fig.savefig(figpath.with_suffix(f), bbox_inches="tight", pad_inches=0.01)
        for f in formats
    ]
