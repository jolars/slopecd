from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from libsvmdata import fetch_libsvm
from pyprojroot import here
from scipy import stats

from slope import plot_utils
from slope.plot_utils import _plot_legend_apart, configure_plt
from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope

save_fig = True

X, y = fetch_libsvm("rcv1.binary")

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

cmap = plt.get_cmap("viridis")
cmap = cmap(np.linspace(0.0, 1.0, 10))
alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 10

verbose = True
max_epochs = 10000
tol = 1e-10
gap_freq = 10
max_time = np.inf

freqs = np.arange(1, 10)

betas = list()
primals = list()
gaps = list()
times = list()
for k in freqs:
    beta_cd, _, primals_cd, gap_cd, time_cd, _ = hybrid_cd(
        X,
        y,
        alphas,
        max_epochs=max_epochs,
        verbose=verbose,
        tol=tol,
        pgd_freq=k,
        gap_freq=gap_freq,
        max_time=max_time,
    )
    betas.append(beta_cd)
    primals.append(primals_cd)
    gaps.append(gap_cd)
    times.append(time_cd)

# Time vs. suboptimality
plt.close("all")

plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots(
    1,
    1,
    figsize=[plot_utils.HALF_WIDTH * 1.2, plot_utils.HALF_WIDTH * 1],
    layout="constrained",
)
minimum = np.min(primals[4])
for k in freqs:
    ax.semilogy(
        times[k - 1],
        np.array(primals[k - 1]) - minimum,
        label="%s" % k,
        color=cmap[k],
    )

ax.set_ylabel(r"$P(\beta) - P(\beta^*)$")
ax.set_xlabel("Time (s)")
ax.set_ylim(1e-10, None)
ax.set_xlim(0, 2)


if save_fig:
    fig.savefig(
        plot_utils.fig_path("pgd_freq.pdf"), bbox_inches="tight", pad_inches=0.05
    )
    plot_utils.plot_legend_apart(
        ax, plot_utils.fig_path("pgd_freq_legend.pdf"), ncol=1, title="PGD Frequency"
    )
else:
    plt.show(block=False)
