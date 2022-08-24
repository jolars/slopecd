
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import stats

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope, slope_threshold, prox_slope
from slope.clusters import get_clusters, update_cluster
from slope.utils import ConvergenceMonitor

dir_results = "../../figures/"


def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color="grey")


def cd(X, y, alphas, max_iter, beta0):
    n_samples = X.shape[0]
    beta = beta0.copy()
    R = y - X @ beta

    beta1s = []
    beta2s = []

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-12, 1, np.inf, True, intercept_column=False
    )

    c, cluster_ptr, cluster_ind, nb_cluster = get_clusters(beta)

    for it in range(max_iter):
        j = 0
        while j < nb_cluster:
            beta1s.append(beta[0])
            beta2s.append(beta[1])

            cluster = cluster_ind[cluster_ptr[j] : cluster_ptr[j + 1]]
            sign_w = np.sign(beta[cluster]) if c[j] != 0 else np.ones(len(cluster))
            sum_X = X[:, cluster] @ sign_w
            L_j = sum_X.T @ sum_X / n_samples
            c_old = abs(c[j])
            x = c_old + (sum_X.T @ R) / (L_j * n_samples)
            beta_tilde, ind_new = slope_threshold(
                x, alphas / L_j, cluster_ptr, c, nb_cluster, j)

            beta[cluster] = beta_tilde * sign_w
            if c_old != beta_tilde:
                R += (c_old - beta_tilde) * sum_X

            ind_old = j
            _ = update_cluster(
                c, cluster_ptr, cluster_ind, nb_cluster, abs(beta_tilde),
                ind_old, ind_new
            )

            j += 1

        converged = monitor.check_convergence(beta, 0.0, it)
        if converged:
            break
    return beta1s, beta2s


def hybrid_cd(X, y, alphas, max_iter, beta0):
    n_samples = X.shape[0]
    beta = beta0

    R = y - X @ beta

    beta1s = []
    beta2s = []

    L = norm(X, ord=2) ** 2 / n_samples

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-12, 1, np.inf, True, intercept_column=False
    )

    c, cluster_ptr, cluster_ind, nb_cluster = get_clusters(beta)

    for it in range(max_iter):
        if it % 5 == 0:
            beta1s.append(beta[0])
            beta2s.append(beta[1])
            beta = prox_slope(beta + X.T @ R / (L * n_samples), alphas)
            R = y - X @ beta
        else:
            j = 0

            while j < nb_cluster:
                beta1s.append(beta[0])
                beta2s.append(beta[1])

                c, cluster_ptr, cluster_ind, nb_cluster = get_clusters(beta)

                cluster = cluster_ind[cluster_ptr[j] : cluster_ptr[j + 1]]
                sign_w = np.sign(beta[cluster]) if c[j] != 0 else np.ones(len(cluster))
                sum_X = X[:, cluster] @ sign_w
                L_j = sum_X.T @ sum_X / n_samples
                c_old = abs(c[j])
                x = c_old + (sum_X.T @ R) / (L_j * n_samples)
                beta_tilde, ind_new = slope_threshold(
                    x, alphas / L_j, cluster_ptr, c, nb_cluster, j)

                beta[cluster] = beta_tilde * sign_w
                if c_old != beta_tilde:
                    R += (c_old - beta_tilde) * sum_X

                ind_old = j
                _ = update_cluster(
                    c, cluster_ptr, cluster_ind, nb_cluster,
                    abs(beta_tilde), ind_old, ind_new
                )

                j += 1
        converged = monitor.check_convergence(beta, 0.0, it)
        if converged:
            break
    return beta1s, beta2s


def pgd(X, y, alphas, max_iter, beta0):

    beta = beta0
    R = y - X @ beta

    beta1s = []
    beta2s = []

    L = norm(X, ord=2) ** 2 / n_samples

    monitor = ConvergenceMonitor(
        X, y, alphas, 1e-12, 1, np.inf, True, intercept_column=False
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


np.random.seed(520)
n_samples = 20
p = 2

X = np.random.rand(n_samples, p)
beta_true = np.array([0.15, -0.72])

y = X @ beta_true

randnorm = stats.norm(loc=0, scale=1)
q = 0.1
reg = 0.01

alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq * reg

beta0 = np.array([-0.8, 0.8])

beta1s_cd, beta2s_cd = cd(X, y, alphas, 20, beta0)
beta1s_hybrid, beta2s_hybrid = hybrid_cd(X, y, alphas, 20, beta0)
beta1s_pgd, beta2s_pgd = pgd(X, y, alphas, 50, beta0)
betas = (
    (beta1s_cd, beta2s_cd), (beta1s_hybrid, beta2s_hybrid), (beta1s_pgd, beta2s_pgd))
beta_star, primals_star, gaps_star, _, _ = prox_grad(
    X, y, alphas, max_epochs=1000, verbose=True, fit_intercept=False
)
beta1 = np.linspace(-1.0, 1.0, 40)
beta2 = np.linspace(-1.0, 1.0, 40)

z = np.ndarray((40, 40))

for i in range(40):
    for j in range(40):
        betax = np.array([beta1[i], beta2[j]])
        r = X @ betax - y
        theta = -r / max(1, dual_norm_slope(X, r, alphas))
        primal = 0.5 * norm(r) ** 2 + np.sum(alphas *
                                             np.sort(np.abs(betax))[::-1])
        dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
        gap = primal - dual
        z[j][i] = gap


labels = ['cd', 'hybrid', 'pgd']

fig, ax = plt.subplots(1, 3, sharex=False, sharey=True, figsize=[10.67, 2.5],)
for i in range(3):
    ax[i].contour(beta1, beta2, z, levels=40)
    x_vals = np.array(ax[i].get_xlim())
    y_vals = - x_vals
    ax[i].plot(x_vals, y_vals, "--", color="grey")
    x_vals = np.array(ax[i].get_xlim())
    y_vals = x_vals
    ax[i].plot(x_vals, y_vals, "--", color="grey")
    ax[i].set_title(labels[i])
    ax[i].plot(beta_star[0], beta_star[1], color="red", marker="x", markersize=16)
    ax[i].plot(
        betas[i][0], betas[i][1], marker="o",
        color="blue", label=labels[i], markersize=4)
fig.savefig(dir_results + 'illustration_solvers.pdf')
fig.show()

# plt.clf()
# plt.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
# plt.plot(np.arange(maxit), gaps)
# plt.show(block=False)
