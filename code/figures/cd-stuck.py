import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.linalg import norm
from numpy.random import default_rng
from scipy import stats

from slope.solvers import hybrid_cd
from slope.utils import lambda_sequence, sl1_norm

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

beta_star, _, _, _, _ = hybrid_cd(X, y, lambdas, False)

betas = np.linspace(-0.05, 0.4, m)

f = np.zeros((m, m))

plt.close("all")

for i in range(m):
    for j in range(m):
        beta = np.hstack((betas[i], betas[j]))
        f[i, j] = (0.5 / n) * norm(y - X @ beta) ** 2 + sl1_norm(beta, lambdas)

b1s, b2s = np.meshgrid(betas, betas)

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
ax.set_aspect("equal")

levels = np.geomspace(np.min(f), np.max(f), 15)
ax.contour(betas, betas, f, levels=levels, colors="darkgrey")

# optimum
ax.plot(
    beta_star[0], beta_star[1], color="darkorange", marker="x", markersize=7, mew=2
)

# where we are stuck
xlim = ax.get_xlim()
ylim = ax.get_ylim()

beta_step = np.array([0.2015, 0.2015])

ax.hlines(beta_step[0], xlim[0], xlim[1], linestyle="--")
ax.vlines(beta_step[1], ylim[0], ylim[1], linestyle="--")
ax.plot(beta_step[0], beta_step[1], color="black", marker=".", markersize=10)

# labels
ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")

plt.rcParams["text.usetex"] = True
plt.savefig("../figures/naive-cd-stuck.pdf", bbox_inches="tight", pad_inches=0.01)

plt.show(block=False)
