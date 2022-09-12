import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.linalg import norm
from numpy.random import default_rng
from scipy import stats

from slope.clusters import get_clusters, update_cluster
from slope.solvers import prox_grad
from slope.utils import ConvergenceMonitor, dual_norm_slope, prox_slope, slope_threshold

dir_results = "../figures/"
savefig = True


def cd(X, y, alphas, max_iter, beta0, verbose=False):
    n_samples = X.shape[0]
    beta = beta0.copy()
    R = y - X @ beta

    betas_epoch = []
    betas_cd = []

    betas_epoch.append(beta.copy())
    betas_cd.append(beta.copy())

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-10, 1, np.inf, verbose, intercept_column=False
    )
    c, c_ptr, c_ind, c_perm, n_clusters = get_clusters(beta)

    for it in range(max_iter):
        j = 0
        beta_prev = beta.copy()
        while j < n_clusters:
            c, c_ptr, c_ind, c_perm, n_clusters = get_clusters(beta)

            cluster = c_ind[c_ptr[j] : c_ptr[j + 1]]
            sign_w = np.sign(beta[cluster]) if c[j] != 0 else np.ones(len(cluster))
            x_tilde = X[:, cluster] @ sign_w
            L_j = x_tilde.T @ x_tilde / n_samples
            c_old = c[c_perm[j]]
            x = c_old + (x_tilde.T @ R) / (L_j * n_samples)
            z, ind_new = slope_threshold(
                x, alphas / L_j, c_ptr, c_perm, c, n_clusters, j
            )

            c_new = np.abs(z)

            beta[cluster] = z * sign_w
            if c_old != z:
                R += (c_old - z) * x_tilde

            ind_old = j
            n_clusters = update_cluster(
                c,
                c_ptr,
                c_ind,
                c_perm,
                n_clusters,
                c_new,
                c_old,
                ind_old,
                ind_new,
                beta,
                X,
                np.zeros([X.shape[0], 2]),
                np.zeros(1),
                use_reduced_X=False,
            )

            j += 1

            betas_cd.append(beta.copy())

        betas_epoch.append(beta.copy())

        converged = monitor.check_convergence(beta, 0.0, it)
        if converged:
            break

        # stop if we get stuck
        if np.array_equal(beta, beta_prev):
            break

    return np.vstack(betas_epoch), np.vstack(betas_cd)


def hybrid_cd(X, y, alphas, max_iter, beta0, verbose=False):
    n_samples = X.shape[0]
    beta = beta0

    R = y - X @ beta

    betas = []
    beta_pgd = []
    beta_cd = []

    beta_tmp = []

    L = norm(X, ord=2) ** 2 / n_samples

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-10, 1, np.inf, verbose, intercept_column=False
    )

    c, c_ptr, c_ind, c_perm, n_clusters = get_clusters(beta)

    for it in range(max_iter):
        if it % 5 == 0:
            betas.append(beta.copy())
            if it > 0:
                beta_cd.append(beta_tmp.copy())
                beta_tmp.clear()
            beta_prev = beta.copy()
            beta = prox_slope(beta + X.T @ R / (L * n_samples), alphas / L)
            R = y - X @ beta
            beta_pgd.append(np.array([beta_prev, beta.copy()]))
            beta_tmp.append(beta.copy())
            betas.append(beta.copy())
            c, c_ptr, c_ind, c_perm, n_clusters = get_clusters(beta)
        else:
            j = 0

            while j < n_clusters:
                if j >= n_clusters:
                    continue

                cluster = c_ind[c_ptr[j] : c_ptr[j + 1]]
                sign_beta = (
                    np.sign(beta[cluster])
                    if c[c_perm[j]] != 0
                    else np.ones(len(cluster))
                )

                x_tilde = X[:, cluster] @ sign_beta
                L_j = x_tilde.T @ x_tilde / n_samples
                c_old = c[c_perm[j]]

                x = c_old + (x_tilde.T @ R) / (L_j * n_samples)
                z, ind_new = slope_threshold(
                    x, alphas / L_j, c_ptr, c_perm, c, n_clusters, j
                )

                c_new = np.abs(z)
                beta[cluster] = z * sign_beta

                if c_old != z:
                    R += (c_old - z) * x_tilde

                ind_old = j

                n_clusters = update_cluster(
                    c,
                    c_ptr,
                    c_ind,
                    c_perm,
                    n_clusters,
                    c_new,
                    c_old,
                    ind_old,
                    ind_new,
                    beta,
                    X,
                    np.zeros([X.shape[0], 2]),
                    np.zeros(1),
                    use_reduced_X=False,
                )

                beta_tmp.append(beta.copy())

                j += 1

            betas.append(beta.copy())

        converged = monitor.check_convergence(beta, 0.0, it)

        if converged:
            if it % 5 != 0:
                beta_cd.append(beta_tmp.copy())
            break

    return np.vstack(betas), beta_pgd, beta_cd


def pgd(X, y, alphas, max_iter, beta0, verbose=False):

    beta = beta0
    R = y - X @ beta

    beta1s = []
    beta2s = []

    L = norm(X, ord=2) ** 2 / n_samples

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-10, 1, np.inf, verbose, intercept_column=False
    )

    for it in range(max_iter):
        beta1s.append(beta[0])
        beta2s.append(beta[1])
        beta = prox_slope(beta + X.T @ R / (L * n_samples), alphas / L)
        R = y - X @ beta
        converged = monitor.check_convergence(beta, 0.0, it)
        if converged:
            break

    return beta1s, beta2s


n_samples = 10

rng = default_rng(3)

cov = [[1, 0.85], [0.85, 1]]
X = rng.multivariate_normal(size=n_samples, mean=[0.0, 0.0], cov=cov)
beta_true = np.array([0.2, -0.2])

y = X @ beta_true

randnorm = stats.norm(loc=0, scale=1)
q = 0.4
reg = 0.3

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))
alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)
alphas = alpha_max * alphas_seq * reg

beta0 = np.array([-0.8, 0.3])  # try to make beta slope have 1 cluster

n_it = 200

betas_hybrid, betas_hybrid_pgd, betas_hybrid_cd = hybrid_cd(X, y, alphas, n_it, beta0)

betas_cd, betas_cd_cd = cd(X, y, alphas, n_it, beta0)
beta1s_pgd, beta2s_pgd = pgd(X, y, alphas, n_it, beta0)
betas = (
    (betas_cd_cd[:, 0], betas_cd_cd[:, 1]),
    (betas_hybrid[:, 0], betas_hybrid[:, 1]),
    (beta1s_pgd, beta2s_pgd),
)
beta_star, primals_star, gaps_star, _, _ = prox_grad(
    X, y, alphas, max_epochs=1000, verbose=False, fit_intercept=False
)

xmin = np.min([np.min(betas[i][0]) for i in range(3)]) - 0.1
xmax = np.max([np.max(betas[i][0]) for i in range(3)]) + 0.1
ymin = np.min([np.min(betas[i][1]) for i in range(3)]) - 0.1
ymax = np.max([np.max(betas[i][1]) for i in range(3)]) + 0.1

beta1 = np.linspace(xmin, xmax, 40)
beta2 = np.linspace(ymin, ymax, 40)

z = np.ndarray((40, 40))

for i in range(40):
    for j in range(40):
        betax = np.array([beta1[i], beta2[j]])
        r = X @ betax - y
        theta = -r / max(1, dual_norm_slope(X, r, alphas))
        primal = 0.5 * norm(r) ** 2 + np.sum(alphas * np.sort(np.abs(betax))[::-1])
        dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
        z[j][i] = primal


labels = ["CD", "Hybrid", "PGD"]

plt.close("all")

fig, axarr = plt.subplots(
    1, 3, sharey=True, sharex=True, figsize=(6.2, 3), constrained_layout=True
)

for i in range(3):
    ax = axarr[i]
    ax.contour(beta1, beta2, z, levels=30, colors="darkgrey")
    x_vals = np.array(ax.get_xlim())
    y_vals = -x_vals
    ax.plot(x_vals, y_vals, ":", color="grey")
    x_vals = np.array(ax.get_xlim())
    y_vals = x_vals
    ax.plot(x_vals, y_vals, ":", color="grey")
    ax.set_title(labels[i])
    ax.plot(
        beta_star[0], beta_star[1], color="darkorange", marker="x", markersize=10, mew=2
    )

    if i == 0:
        linestyle = "-"
        marker = ""
    elif i == 1:
        linestyle = ""
        marker = "."
    else:
        linestyle = "--"
        marker = "."

    ax.plot(
        betas[i][0],
        betas[i][1],
        marker=marker,
        color="black",
        label=labels[i],
        linestyle=linestyle,
        markersize=6,
    )

    if i == 0:
        # Plot epochs as points
        ax.plot(
            betas_cd[:, 0],
            betas_cd[:, 1],
            marker=".",
            color="black",
            label=labels[i],
            linestyle="",
            markersize=8,
        )

    if i == 1:
        # Plot separate linestyles for CD and hybrid steps
        lc_cd = LineCollection(betas_hybrid_cd, color="black")
        lc_pgd = LineCollection(betas_hybrid_pgd, linestyle="--", color="black")
        ax.add_collection(lc_cd)
        ax.add_collection(lc_pgd)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\beta_1$")
    if i == 0:
        ax.set_ylabel(r"$\beta_2$")

if savefig:
    plt.rcParams["text.usetex"] = True
    fig.savefig(
        dir_results + "illustration_solvers.pdf", bbox_inches="tight", pad_inches=0.01
    )
    fig.savefig(
        dir_results + "illustration_solvers.svg", bbox_inches="tight", pad_inches=0.01
    )

plt.show(block=False)
