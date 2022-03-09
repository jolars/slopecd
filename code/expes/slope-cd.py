from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
from scipy import stats

from slope.solvers import oracle_cd, prox_grad
from slope.utils import dual_norm_slope


def get_clusters(w):
    unique, indices, counts = np.unique(
        np.abs(w), return_inverse=True, return_counts=True
    )

    clusters = [[] for _ in range(len(unique))]
    for i in range(len(indices)):
        clusters[indices[i]].append(i)
    return clusters[::-1], counts[::-1], indices[::-1], unique[::-1]


# def lambda_sums(lambdas, C, C_size, C_start, C_end, c, j):
#     C_size_j = C_size[j]

#     lo_sums = []
#     up_sums = []

#     for k in range(len(C)):
#         if k == j:
#             continue

#         mod = C_size_j if k > j else 0

#         # check upper end of cluster
#         up_start = C_start[k] - mod
#         up_end = C_start[k] + C_size_j - mod

#         # check lower end of cluster
#         lo_start = C_end[k] - mod
#         lo_end = C_end[k] + C_size_j - mod

#         lo_sum = sum(lambdas[lo_start:lo_end])
#         up_sum = sum(lambdas[up_start:up_end])

#         lo_sums.extend([lo_sum])
#         up_sums.extend([up_sum])

#     if ~any(np.delete(C, j) == 0):
#         lo_sums.extend([0.0])
#         up_sums.extend([sum(lambdas[::-1][range(C_size_j)])])

#     return (
#         np.array(lo_sums),
#         np.array(up_sums),
#         np.flip(np.unique(np.hstack((lo_sums, up_sums)))),
#     )


def slope_threshold(x, lambdas, C, C_start, C_end, c, j):

    A = C[j]
    cluster_size = len(A)

    # zero_cluster_size = 0 if c[-1] != 0 else len(C[-1])

    zero_lambda_sum = np.sum(lambdas[::-1][range(cluster_size)])

    if np.abs(x) < zero_lambda_sum:
        return 0.0

    lo = 0.0
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
            return np.sign(x) * (np.abs(x) - hi)

        if abs(x) > lo + c[k] and abs(x) < hi + c[k]:
            # we are in a cluster
            return np.sign(x) * c[k]

    return np.sign(x) * (np.abs(x) - lo)


np.random.seed(10)
n = 10
p = 2

X = np.random.rand(n, p)
# w = np.random.rand(p)
# beta = np.array([0.5, 0.5, -0.7, 0.2])
beta = np.array([0.5, -0.5])
y = X @ beta

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
q = 0.8

lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
lambda_max = dual_norm_slope(X, y, lambdas)
lambdas = lambda_max * lambdas * 0.5

max_it = 20

beta = np.zeros(p)

C, C_size, C_indices, c = get_clusters(beta)

r = X @ beta - y
g = X.T @ r

gaps = []
primals = []
duals = []

maxit = 100

for it in range(maxit):
    j = 0

    C, C_size, C_indices, c = get_clusters(beta)
    C_end = np.cumsum(C_size)
    C_start = C_end - C_size
    C_ord = np.arange(len(C))

    print(f"Iter: {it + 1}")
    print(f"\tbeta: {beta}")

    while j < len(C):
        A = C[j].copy()
        lambdas_j = lambdas[C_start[j] : C_end[j]]

        print(f"\tj: {j}, A: {A}, c_j: {c[j]:.2e}")

        grad_A = X[:, A].T @ r

        # see if we need to split up the cluster
        new_cluster = []
        new_cluster_tmp = []
        grad_sum = 0
        lambda_sum = 0

        # check if the clusters should split
        grad_order = np.argsort(np.abs(grad_A))[::-1]

        if len(A) > 1:
            for k, ind in enumerate(grad_order):
                grad_sum += abs(grad_A[ind])
                lambda_sum += lambdas_j[k]
                new_cluster_tmp.extend([A[ind]])
                if grad_sum > lambda_sum:
                    new_cluster = new_cluster_tmp
                    if len(new_cluster) < len(A):
                        C[j] = [Cjk for Cjk in C[j] if Cjk not in new_cluster]
                    break

        if len(new_cluster) > 0 and len(new_cluster) < len(A):
            # the cluster splits
            C = C[0:j] + [new_cluster] + [C[j]] + C[(j + 1) :]

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
            A = new_cluster.copy()

        c_old = c[j]
        B = list(set(range(p)) - set(A))
        c_wo_j = np.delete(c, j)

        # s = np.sign(beta[A])
        s = np.sign(-g[A])
        s = np.ones(len(s)) if all(s == 0) else s
        H = s.T @ X[:, A].T @ X[:, A] @ s
        x = (y - X[:, B] @ beta[B]).T @ X[:, A] @ s
        # x = c[j] - r.T @ X[:, A] @ s / H

        # lo_sums, up_sums, sums = lambda_sums(lambdas, C, C_size, C_start, C_end, c, j)

        beta_tilde = slope_threshold(x / H, lambdas / H, C, C_start, C_end, c, j)
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

    theta = -r / max(1, dual_norm_slope(X, r, lambdas))

    primal = 0.5 * norm(r) ** 2 + np.sum(lambdas * np.sort(np.abs(beta))[::-1])
    dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
    gap = primal - dual

    print(f"\tloss: {primal:.2e}, gap: {gap:.2e}")

    primals.append(primal)
    duals.append(dual)
    gaps.append(gap)


beta_star, primals_star, gaps_star, theta = prox_grad(
    X, y, lambdas / n, max_iter=1000, n_cd=0, verbose=False
)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
plt.plot(np.arange(maxit), gaps)
ax.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
plt.show()
