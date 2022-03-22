from random import sample
from timeit import default_timer as timer

import numpy as np
from numba import njit
from numpy.linalg import norm

from slope.clusters import Clusters
from slope.utils import dual, dual_norm_slope, primal, prox_slope


# thi is basically the proximal operator with a few steps removed
@njit
def find_splits(x, lam):
    x = np.abs(x)
    ord = np.flip(np.argsort(x))
    x = x[ord]

    p = len(x)

    s = np.empty(p)
    w = np.empty(p)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = x[i] - lam[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    return ord[idx_i[0]: (idx_j[0] + 1)]


def slope_threshold(x, lambdas, clusters, j):

    A = clusters.inds[j]
    cluster_size = len(A)

    zero_lambda_sum = np.sum(lambdas[::-1][range(cluster_size)])

    if np.abs(x) < zero_lambda_sum:
        return 0.0, len(clusters.coefs) - 1

    lo = zero_lambda_sum
    hi = zero_lambda_sum

    # TODO(JL): This can and should be done much more efficiently, using
    # kind of binary search to find the right interval

    k = 0
    mod = 0

    for k in range(len(clusters.coefs)):
        if k == j:
            continue

        # adjust pointers if we are ahead of the cluster in order
        mod = cluster_size if k > j else 0

        # check upper end of cluster
        hi_start = clusters.starts[k] - mod
        hi_end = clusters.starts[k] + cluster_size - mod

        # check lower end of cluster
        lo_start = clusters.ends[k] - mod
        lo_end = clusters.ends[k] + cluster_size - mod

        lo = sum(lambdas[lo_start:lo_end])
        hi = sum(lambdas[hi_start:hi_end])

        cluster_k = k - 1 if k > j else k

        if abs(x) > hi + clusters.coefs[k]:
            # we must be between clusters
            return x - np.sign(x) * hi, cluster_k
        elif abs(x) >= lo + clusters.coefs[k]:
            # we are in a cluster
            return np.sign(x) * clusters.coefs[k], cluster_k

    cluster_k = k - 1 if k > j else k

    # we are between clusters
    return x - np.sign(x) * lo, cluster_k


def proxsplit_cd(X, y, lambdas, max_epochs=100, tol=1e-10, split_freq=1, verbose=False):
    n, p = X.shape

    beta = np.zeros(p)
    theta = np.zeros(n)

    r = -y
    g = (X.T @ r) / n

    times = []
    time_start = timer()

    L = norm(X, ord=2)**2 / n

    primals, duals, gaps = [], [], []

    primals.append(norm(y) ** 2 / (2 * n))
    duals.append(0)
    gaps.append(primals[0])
    times.append(timer() - time_start)

    epoch = 0

    features_seen = 0

    clusters = Clusters(beta)

    while epoch < max_epochs:
        r = X @ beta - y

        theta = -r / n
        theta /= max(1, dual_norm_slope(X, theta, lambdas))

        primal = (0.5 / n) * norm(r) ** 2 + np.sum(
            lambdas * np.sort(np.abs(beta))[::-1]
        )
        dual = (0.5 / n) * (norm(y) ** 2 - norm(y - theta * n) ** 2)
        gap = primal - dual

        primals.append(primal)
        duals.append(dual)
        gaps.append(gap)
        times.append(timer() - time_start)

        if verbose:
            print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")

        if gap < tol:
            break

        while features_seen < p:
            j = sample(range(len(clusters.coefs)), 1)[0]

            C = clusters.inds[j]
            c = clusters.coefs[j]
            lambdas_j = lambdas[clusters.starts[j]: clusters.ends[j]]

            g = (X[:, C].T @ r) / n

            if len(C) > 1 and epoch % split_freq == 0:
                # check if clusters should split and if so how
                x = beta[C] - g / L
                split = find_splits(x, lambdas_j / L)

                if len(split) < len(C):
                    C = [C[i] for i in split]
                    clusters.split(j, C)
                    g = g[split]

            s = -np.sign(g)

            sum_X = X[:, C] @ s
            L_j = (sum_X.T @ sum_X) / n
            x = c - (s.T @ g) / L_j

            beta_tilde, new_ind = slope_threshold(
                x, lambdas / L_j, clusters, j)

            clusters.update(j, new_ind, abs(beta_tilde))

            beta[C] = beta_tilde * s

            r -= (c - beta_tilde) * sum_X
            # r = X @ beta - y

            features_seen += len(C)

        epoch += 1

        features_seen -= p

    return beta, primals, gaps, theta, times
