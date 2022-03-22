from random import sample

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from numba import njit
from numpy.linalg import norm
from scipy import stats

from slope.clusters import Clusters
from slope.solvers.oracle import oracle_cd
from slope.solvers.pgd import prox_grad
from slope.utils import dual_norm_slope


def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color="grey")


# this is basically the simplified proximal operator, but simplified
@njit
def find_splits(x, lam):
    x_sign = np.sign(x)
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

    # for j in range(k):
    #     d = max(w[j], 0.0)
    #     for i in range(idx_i[j], idx_j[j] + 1):
    #         x[i] = d

    # x[ord] = x.copy()
    # x *= x_sign

    return ord[idx_i[0] : (idx_j[0] + 1)]
    # return x


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


n = 100
p = 500

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

y = y - np.mean(y)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5
alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

max_epochs = 1000
tol = 1e-10
split_freq = 1
verbose = True


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq / 5

n, p = X.shape

beta = np.zeros(p)
theta = np.zeros(n)

r = -y
g = (X.T @ r) / n

clusters = Clusters(beta)

L = norm(X, ord=2) ** 2 / n

primals, duals, gaps = [], [], []

epoch = 0

features_seen = 0


# w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
# R[:] = y - X @ w

while epoch < max_epochs:
    r = X @ beta - y

    theta = -r / n
    theta /= max(1, dual_norm_slope(X, theta, lambdas))

    primal = (0.5 / n) * norm(r) ** 2 + np.sum(
        lambdas * np.sort(np.abs(beta))[::-1]
    )
    dual = (0.5 / n) * (norm(y) ** 2 - norm(y - theta * n) ** 2)
    gap = primal - dual

    if gap < tol:
        break

    primals.append(primal)
    duals.append(dual)
    gaps.append(gap)

    if verbose:
        print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")

    while features_seen < p:
        j = sample(range(len(clusters.coefs)), 1)[0]

        C = clusters.inds[j]
        c = clusters.coefs[j]
        lambdas_j = lambdas[clusters.starts[j] : clusters.ends[j]]

        g = (X[:, C].T @ r) / n

        if len(C) > 1 and epoch % split_freq == 0:
            # check if clusters should split and if so how
            x = beta[C] - g / L
            split = find_splits(x, lambdas_j / L)

            if len(split) < len(C):
                C = [C[i] for i in split]
                clusters.split(j, C)
                g = g[split]

        C = clusters.inds[j]
        c = clusters.coefs[j]

        s = -np.sign(g)

        sum_X = X[:, C] @ s
        L_j = (sum_X.T @ sum_X) / n
        x = c - (s.T @ g) / L_j

        beta_tilde, new_ind = slope_threshold(x, lambdas / L_j, clusters, j)

        clusters.update(j, new_ind, abs(beta_tilde))

        beta[C] = beta_tilde * s

        # r -= (c - beta_tilde) * sum_X
        r = X @ beta - y

        features_seen += len(C)

    epoch += 1

    features_seen -= p

beta_star, primals_star, gaps_star, theta_star = prox_grad(
    X, y, lambdas / n, max_epochs=1000, verbose=False
)

plt.clf()
# plt.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
plt.semilogy(np.arange(len(gaps)), gaps)
plt.show(block=False)
