import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import stats

from slope.clusters import Clusters
from slope.solvers import oracle_cd, prox_grad
from slope.utils import dual_norm_slope


def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color="grey")


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

        if abs(x) > hi + clusters.coefs[k]:
            # we must be between clusters
            return x - np.sign(x) * hi, k - mod
        elif abs(x) >= lo + clusters.coefs[k]:
            # we are in a cluster
            return np.sign(x) * clusters.coefs[k], k - mod

    # we are between clusters
    return x - np.sign(x) * lo, k - mod


np.random.seed(10)
n = 10
p = 2

X = np.random.rand(n, p)
beta_true = np.array([0.5, -0.8])

y = X @ beta_true

randnorm = stats.norm(loc=0, scale=1)
q = 0.8

lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
lambda_max = dual_norm_slope(X, y, lambdas)
lambdas = lambda_max * lambdas * 0.5

beta = np.array([0, 0])

beta1_start = beta[0]
beta2_start = beta[1]

r = X @ beta - y
g = X.T @ r

gaps = []
primals = []
duals = []

beta1s = []
beta2s = []

L = norm(X, ord=2) ** 2

maxit = 10

clusters = Clusters(beta)

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

    while j < len(clusters.coefs):
        beta1s.append(beta[0])
        beta2s.append(beta[1])

        A = clusters.inds[j].copy()
        lambdas_j = lambdas[clusters.starts[j] : clusters.ends[j]]

        print(f"\tj: {j}, C: {clusters.inds[j]}, c_j: {clusters.coefs[j]:.2e}")

        grad_A = X[:, A].T @ r

        # see if we need to split up the cluster
        new_cluster = []
        new_cluster_tmp = []
        grad_sum = 0
        lambda_sum = 0

        # check if clusters should split and if so how
        if len(A) > 1:
            grad_order = np.argsort(np.abs(grad_A))
            possible_split_index = -1
            best_grad_sum = 0
            for i, ind in enumerate(grad_order):
                grad_sum += abs(grad_A[ind])
                lambda_sum += lambdas_j[i]
                if grad_sum >= lambda_sum and grad_sum >= best_grad_sum:
                    possible_split_index = i
                    best_grad_sum = grad_sum
                    break

            print(f"possible split: {possible_split_index}")
            # left_split = [A[i] for i in list(np.where(v_best)[0])]
            if possible_split_index >= 0:
                left_split = grad_order[:(possible_split_index + 1)]
                print(left_split)
                clusters.split(j, left_split)

        A = clusters.inds[j]
        B = list(set(range(p)) - set(A))

        s = np.sign(-g[A])
        H = s.T @ X[:, A].T @ X[:, A] @ s
        x = (y - X[:, B] @ beta[B]).T @ X[:, A] @ s

        beta_tilde, new_ind = slope_threshold(x / H, lambdas / H, clusters, j)

        clusters.update(j, new_ind, abs(beta_tilde))

        beta[A] = beta_tilde * s

        j += 1

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
