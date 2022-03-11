from bisect import bisect_left
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
from scipy import stats

from slope.solvers import oracle_cd, prox_grad
from slope.utils import dual_norm_slope, prox_slope


def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color="grey")


def get_clusters(w):
    unique, indices, counts = np.unique(
        np.abs(w), return_inverse=True, return_counts=True
    )

    clusters = [[] for _ in range(len(unique))]
    for i in range(len(indices)):
        clusters[indices[i]].append(i)
    return clusters[::-1], counts[::-1], indices[::-1], unique[::-1]


def slope_threshold(x, lambdas, C, C_start, C_end, c, j):

    A = C[j]
    cluster_size = len(A)

    # zero_cluster_size = 0 if c[-1] != 0 else len(C[-1])

    zero_lambda_sum = np.sum(lambdas[::-1][range(cluster_size)])

    if np.abs(x) < zero_lambda_sum:
        return 0.0

    lo = zero_lambda_sum
    hi = zero_lambda_sum

    # TODO(JL): This can and should be done much more efficiently, using
    # kind of binary search to find the right interval
    for k in range(len(C)):
        if k == j:
            continue

        # adjust C_start and C_end since we treat current cluster as variable
        mod = cluster_size if k > j else 0

        # check upper end of cluster
        hi_start = C_start[k] - mod
        hi_end = C_start[k] + cluster_size - mod

        # check lower end of cluster
        lo_start = C_end[k] - mod
        lo_end = C_end[k] + cluster_size - mod

        lo = sum(lambdas[lo_start:lo_end])
        hi = sum(lambdas[hi_start:hi_end])

        if abs(x) > hi + c[k]:
            # we must be between clusters
            # return np.sign(x) * (np.abs(x) - hi)
            return x - np.sign(x)*hi
        elif abs(x) >= lo + c[k]:
            # we are in a cluster
            return np.sign(x) * c[k]

    # return np.sign(x) * (np.abs(x) - lo)
    return x - np.sign(x)*lo


np.random.seed(10)
n = 10
p = 2

X = np.random.rand(n, p)
beta_true = np.array([0.8, -0.8])
y = X @ beta_true

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
q = 0.8

lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
lambda_max = dual_norm_slope(X, y, lambdas)
lambdas = lambda_max * lambdas * 0.5

max_it = 20

beta = np.array([0.0, 0.7])

beta1_start = beta[0]
beta2_start = beta[1]

C, C_size, C_indices, c = get_clusters(beta)

r = X @ beta - y
g = X.T @ r

gaps = []
primals = []
duals = []

beta1s = []
beta2s = []

L = norm(X, ord=2) ** 2

maxit = 10

for it in range(maxit):
    j = 0

    print(f"Iter: {it + 1}")
    print(f"\tbeta: {beta}")

    r = X @ beta - y
    theta = -r / max(1, dual_norm_slope(X, r, lambdas))

    primal = 0.5 * norm(r) ** 2 + np.sum(lambdas * np.sort(np.abs(beta))[::-1])
    dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
    gap = primal - dual

    print(f"\tloss: {primal:.2e}, gap: {gap:.2e}")

    primals.append(primal)
    duals.append(dual)
    gaps.append(gap)

    C, C_size, C_indices, c = get_clusters(beta)
    C_end = np.cumsum(C_size)
    C_start = C_end - C_size
    C_ord = np.arange(len(C))

    while j < len(C):
        beta1s.append(beta[0])
        beta2s.append(beta[1])

        A = C[j].copy()
        lambdas_j = lambdas[C_start[j] : C_end[j]]

        print(f"\tj: {j}, A: {A}, c_j: {c[j]:.2e}")

        grad_A = X[:, A].T @ r

        # see if we need to split up the cluster
        new_cluster = []
        new_cluster_tmp = []
        grad_sum = 0
        lambda_sum = 0

        # check if clusters should split and if so how
        if len(A) > 1:
            if np.any(c > 0):
                h0 = 0.01 * np.min(np.diff(np.hstack((0, c))))
            else:
                h0 = 0.01  # doesn't matter what we choose

            possible_directions = list(product([0, 1, -1], repeat=len(A)))
            del possible_directions[0]  # remove 0-direction

            v_best = possible_directions[0]
            dir_deriv = 1e8

            # search all directions for best direction
            for i in range(len(possible_directions)):
                v = possible_directions[i]
                v /= norm(v)  # normalize direction

                # smallest epsilon such that current clustering is maintained
                idx = np.flip(np.argsort(np.abs(beta + h0 * v)))
                sgn = np.sign(beta + h0 * v)

                d = np.dot(v, grad_A) + np.sum(lambdas_j[idx] * v * sgn)

                if d < dir_deriv:
                    dir_deriv = d
                    v_best = v.copy()

            new_clusters, _, _, _ = get_clusters(beta[A] + h0 * v_best)
            new_cluster = new_clusters[0]

            if len(new_clusters) > 1:
                # the cluster splits
                C = C[0:j] + new_clusters + C[(j + 1) :]

                above = C_ord > C_ord[j]

                C_ord[above] += 1
                C_ord = np.r_[C_ord[range(j)], C_ord[j], C_ord[j] + 1, C_ord[j + 1 :]]
                C_start = np.r_[
                    C_start[range(j)],
                    C_start[j],
                    C_start[j] + len(new_cluster),
                    C_start[(j + 1) :],
                ]
                C_end = np.r_[
                    C_end[range(j)],
                    C_start[j] + len(new_cluster),
                    C_end[j],
                    C_end[(j + 1) :],
                ]
                C_size = np.r_[
                    C_size[range(j)],
                    len(new_cluster),
                    C_size[j] - len(new_cluster),
                    C_size[(j + 1) :],
                ]
                c = np.r_[c[range(j)], c[j], c[j], c[j + 1 :]]
                A = new_clusters[0]

        c_old = c[j]
        B = list(set(range(p)) - set(A))

        # s = np.sign(beta[A])
        s = np.sign(-g[A])
        s = np.ones(len(s)) if all(s == 0) else s
        H = s.T @ X[:, A].T @ X[:, A] @ s
        x = (y - X[:, B] @ beta[B]).T @ X[:, A] @ s / H

        # lo_sums, up_sums, sums = lambda_sums(lambdas, C, C_size, C_start, C_end, c, j)

        beta_tilde = slope_threshold(x, lambdas/H, C, C_start, C_end, c, j)
        c[j] = np.abs(beta_tilde)
        beta[A] = beta_tilde * s

        # print(f"\t\tnew_c: {beta_tilde:.2e}")

        j += 1

        C, C_size, C_indices, c = get_clusters(beta)
        C_end = np.cumsum(C_size)
        C_start = C_end - C_size
        C_ord = np.arange(len(C))

        r = X @ beta - y
        g = X.T @ r

    r = X @ beta - y

beta_star, primals_star, gaps_star, theta_star = prox_grad(
    X, y, lambdas / n, max_iter=1000, n_cd=0, verbose=False
)

beta1 = np.linspace(-0.8, 0.8, 20)
beta2 = np.linspace(-0.8, 0.8, 20)

z = np.ndarray((20, 20))

for i in range(20):
    for j in range(20):
        betax = np.array([beta1[i], beta2[j]])
        r = X @ betax - y
        theta = -r / max(1, dual_norm_slope(X, r, lambdas))
        primal = 0.5 * norm(r) ** 2 + np.sum(lambdas * np.sort(np.abs(betax))[::-1])
        dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
        gap = primal - dual
        z[j][i] = gap

plt.clf()
plt.contour(beta1, beta2, z, levels=20)
abline(1, 0)
abline(-1, 0)
plt.plot(beta_star[0], beta_star[1], color="red", marker="x", markersize=16)
plt.plot(beta1s, beta2s, marker="o", color="black")
plt.show(block=False)

# plt.clf()
# plt.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
# plt.plot(np.arange(maxit), gaps)
# plt.show(block=False)
