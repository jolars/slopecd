import matplotlib.pyplot as plt
import numpy as np
from libsvmdata import fetch_libsvm
from scipy import stats

from slope.solvers import hybrid_cd
from slope.utils import dual_norm_slope
from slope.plot_utils import configure_plt, _plot_legend_apart

fig_dir = "../../figures/"
configure_plt()
X, y = fetch_libsvm("rcv1.binary")

randnorm = stats.norm(loc=0, scale=1)
q = 0.5

cmap = plt.get_cmap('viridis')
cmap = cmap(np.linspace(0., 1.0, 10))
alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 10

max_epochs = 10000
tol = 1e-10

beta_cd, _, primals_cd, gaps_cd, time_cd, _ = hybrid_cd(
    X, y, alphas, max_epochs=2, verbose=True, tol=tol
)
freqs = np.arange(1, 10)

betas = list()
primals = list()
gaps = list()
times = list()
for k in freqs:
    beta_cd, _, primals_cd, gap_cd, time_cd, _ = hybrid_cd(
        X, y, alphas, max_epochs=max_epochs, verbose=True, tol=tol, pgd_freq=k
    )
    betas.append(beta_cd)
    primals.append(primals_cd)
    gaps.append(gap_cd)
    times.append(time_cd)

# Time vs. duality gap
plt.clf()
fig, ax = plt.subplots(layout="constrained")
minimum = np.min(primals[4])
for k in freqs:
    ax.semilogy(
        times[k - 1], np.array(primals[k - 1]) - minimum, label="pgd freq = %s" % k,
        color=cmap[k])

ax.set_ylabel(r"$P(\beta) - P(\beta^*)$")
ax.set_xlabel("Time (s)")
ax.set_title("rcv1")
ax.set_ylim(1e-10, 1)
ax.set_xlim(0, 2)
fig.savefig(fig_dir + "pgd_freq.pdf")
plt.show(block=False)

_plot_legend_apart(ax, fig_dir + "legend_pgd_freq.pdf", ncol=3)
