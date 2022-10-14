import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng

from slope import plot_utils
from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence, sl1_norm

savefig = True

n = 10
p = 2

rng = default_rng(5)

cov = np.array([[1.0, -0.8], [-0.8, 1]])

X = rng.multivariate_normal([0.0, 0.0], cov, size=n)
beta_in = np.array([1, 1])
y = X @ beta_in

m = 401

lambdas = lambda_sequence(X, y, False, reg=0.999, q=0.99)

lambdas = np.array([0.5, 0.1])

beta_star = hybrid_cd(X, y, lambdas, fit_intercept=False)[0]

lims = (-0.05, 0.45)

betas = np.linspace(lims[0], lims[1], m)

f = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        beta = np.hstack((betas[i], betas[j]))
        f[i, j] = (0.5 / n) * norm(y - X @ beta) ** 2 + sl1_norm(beta, lambdas)

b1s, b2s = np.meshgrid(betas, betas)

beta_stuck = 0.2
beta_stuck_2d = np.array([beta_stuck, beta_stuck])
f_stuck = (0.5 / n) * norm(y - X @ beta_stuck_2d) ** 2 + sl1_norm(
    beta_stuck_2d, lambdas
)

# univariate problems
beta_in = np.sort(np.hstack((beta_stuck, np.linspace(lims[0], lims[1], 200))))

f1 = np.zeros(len(beta_in))
f2 = np.zeros(len(beta_in))

for i in range(len(beta_in)):
    beta1 = np.hstack((beta_in[i], beta_stuck))
    beta2 = np.hstack((beta_stuck, beta_in[i]))
    f1[i] = (0.5 / n) * norm(y - X @ beta1) ** 2 + sl1_norm(beta1, lambdas)
    f2[i] = (0.5 / n) * norm(y - X @ beta2) ** 2 + sl1_norm(beta2, lambdas)

fs = (f1, f2)

plt.close("all")
plt.rcParams["text.usetex"] = True
fig = plt.figure(
    figsize=(plot_utils.HALF_WIDTH, plot_utils.FULL_WIDTH), constrained_layout=True
)
ax = fig.add_gridspec(top=0.4, right=0.4).subplots()

# contours
ax.set_aspect("equal")

n_levels = 15
levels = np.zeros(n_levels)
for i in range(n_levels):
    b = beta_star + 0.05 * i
    levels[i] = (0.5 / n) * norm(y - X @ b) ** 2 + sl1_norm(b, lambdas)


ax_beta1 = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_beta2 = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

ax.contour(betas, betas, f, levels=levels, colors="darkgrey")
ax.plot(beta_star[0], beta_star[1], color="darkorange", marker="x", markersize=7, mew=2)

# where we are stuck
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.hlines(beta_stuck, xlim[0], xlim[1], linestyle="dotted", color="darkgrey")
ax.vlines(beta_stuck, ylim[0], ylim[1], linestyle="dotted", color="darkgrey")
ax.plot(beta_stuck, beta_stuck, color="black", marker=".", markersize=6)

# labels
ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")

ax_beta1.tick_params(axis="x", labelbottom=False)
ax_beta1.vlines(beta_stuck, min(f1), max(f1), linestyle="dotted", color="darkgrey")
ax_beta1.plot(beta_in, f1, color="black")
ax_ylim = ax_beta1.get_ylim()
ax_beta1.set_ylabel(r"$P(\beta)$")

ax_beta2.tick_params(axis="y", labelleft=False)
ax_beta2.hlines(beta_stuck, min(f2), max(f2), linestyle="dotted", color="darkgrey")
ax_beta2.plot(f2, beta_in, color="black")
ax_xlim = ax_beta2.get_xlim()
ax_beta2.set_xlabel(r"$P(\beta)$")

if savefig:
    figpath = plot_utils.fig_path("naive-cd-stuck")
    formats = [".svg", ".pdf"]

    [
        fig.savefig(figpath.with_suffix(f), bbox_inches="tight", pad_inches=0.01)
        for f in formats
    ]
else:
    plt.show(block=False)
