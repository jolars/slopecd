import matplotlib.pyplot as plt
import numpy as np
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

    return ord[idx_i[0] : (idx_j[0] + 1)]


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


np.random.seed(10)
n = 100
p = 400

X = np.random.rand(n, p)
beta_true = np.zeros(p)

y = X @ beta_true + np.random.rand(n)

randnorm = stats.norm(loc=0, scale=1)
q = 0.1

lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
lambda_max = dual_norm_slope(X, y, lambdas)
lambdas = lambda_max * lambdas * 0.1

beta = np.zeros(p)

r = X @ beta - y
g = X.T @ r

gaps = []
primals = []
duals = []

L = norm(X, ord=2) ** 2

maxit = 1000

clusters = Clusters(beta)

for it in range(maxit):
    j = 0

    r = X @ beta - y
    theta = -r / max(1, dual_norm_slope(X, r, lambdas))

    primal = 0.5 * norm(r) ** 2 + np.sum(lambdas * np.sort(np.abs(beta))[::-1])
    dual = 0.5 * (norm(y) ** 2 - norm(y - theta) ** 2)
    gap = primal - dual

    print(f"Iter: {it + 1}")
    print(f"\tloss: {primal:.2e}, gap: {gap:.2e}")

    primals.append(primal)
    duals.append(dual)
    gaps.append(gap)

    while j < len(clusters.coefs):
        A = clusters.inds[j].copy()
        lambdas_j = lambdas[clusters.starts[j] : clusters.ends[j]]

        grad_A = X[:, A].T @ r

        # check if clusters should split and if so how
        if len(A) > 1:
            x = beta[A] - X[:, A].T @ r
            # if clusters.coefs[j] == 0:
            #     ind = np.argmax(np.abs(x))
            #     if np.abs(x)[ind] > lambdas_j[0]:
            #         clusters.split(j, [A[ind]])
            #         A = clusters.inds[j]
            # else:
            left_split = find_splits(x, lambdas_j)
            split_ind = [A[i] for i in left_split]
            clusters.split(j, split_ind)

            A = clusters.inds[j]

        B = list(set(range(p)) - set(A))

        s = np.sign(beta[A])
        s = np.ones(len(s)) if np.all(s == 0) else s
        # s = -np.sign(g[A])

        # sum_X = s.T @ X[:, A].T
        # L_j = sum_X @ sum_X.T 
        # c_old = clusters.coefs[j]
        # x = c_old - (sum_X @ r) / (L_j)
        H = s.T @ X[:, A].T @ X[:, A] @ s
        x = (y - X[:, B] @ beta[B]).T @ X[:, A] @ s

        beta_tilde, new_ind = slope_threshold(x /H, lambdas / H, clusters, j)

        clusters.update(j, new_ind, abs(beta_tilde))

        beta[A] = beta_tilde * s
        j += 1

        # r_tmp = r.copy()
        # r_tmp -= (c_old - beta_tilde) * sum_X.T

        r = X @ beta - y
        
        # if not np.allclose(r_tmp, r):
        #     raise ValueError("")

        g = X.T @ r

    r = X @ beta - y

# beta_star, primals_star, gaps_star, theta_star = prox_grad(
#     X, y, lambdas / n, max_iter=1000, n_cd=0, verbose=False
# )

# plt.clf()
# # plt.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
# plt.semilogy(np.arange(maxit), gaps)
# plt.show(block=False)
