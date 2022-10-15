from random import sample
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets import make_correlated_data
from libsvmdata import fetch_libsvm
from numba import njit
from numpy.linalg import norm
from scipy import stats

from slope.clusters import initialize_clusters, update_cluster, slope_threshold
from slope.solvers.oracle import oracle_cd
from slope.solvers.pgd import prox_grad
from slope.utils import dual_norm_slope, prox_slope


# this is basically the proximal operator, but simplified
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

n = 100
p = 1000

dataset = "simulated"
if dataset == "simulated":
    X, y, _ = make_correlated_data(n_samples=n, n_features=p, random_state=0, rho=0.1)
    # X = csc_matrix(X)
else:
    X, y = fetch_libsvm(dataset)

randnorm = stats.norm(loc=0, scale=1)
q = 0.5
alphas_seq = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


L = norm(X, ord=2) ** 2 / n
alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

lambdas = alpha_max * alphas_seq / 5

verbose = True
split_freq = 1
max_epochs = 500
tol = 1e-4
cyclic = True

pgd_freq = 5

do_cluster_updates = True

n, p = X.shape

beta = np.zeros(p)
theta = np.zeros(n)

r = -y
g = (X.T @ r) / n

times = []
time_start = timer()

primals, duals, gaps = [], [], []

primals.append(norm(y) ** 2 / (2 * n))
duals.append(0)
gaps.append(primals[0])
times.append(timer() - time_start)

epoch = 0

features_seen = 0

c, c_ptr, c_ind, n_c = initialize_clusters(beta)

while epoch < max_epochs:
    r = X @ beta - y

    theta = -r / n
    theta /= max(1, dual_norm_slope(X, theta, lambdas))

    primal = (0.5 / n) * norm(r) ** 2 + np.sum(lambdas * np.sort(np.abs(beta))[::-1])
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

    if epoch % pgd_freq == 0:
        beta = prox_slope(beta - (X.T @ r) / (L * n), lambdas / L)
        r[:] = X @ beta - y
        c, c_ptr, c_ind, n_c = initialize_clusters(beta)
    else:
        j = 0

        while True:
            if cyclic:
                if j >= n_c:
                    break
            else:
                if features_seen > p:
                    break
                else:
                    j = sample(range(n_c), 1)[0]

            lambdas_j = lambdas[c_ptr[j] : c_ptr[j + 1]]

            c_ind_j = c_ind[c_ptr[j] : c_ptr[j + 1]].copy()

            g = (X[:, c_ind_j].T @ r) / n

            do_split = False

            # if len(C_j) > 1 and epoch % split_freq == 0:
            #     if any([c_i > 0 for c_i in c]):
            #         h0 = 0.1 * np.abs(np.min(np.diff(np.hstack((0, c)))))
            #     else:
            #         h0 = 0.1  # doesn't matter what we choose

            #     # possible_directions = list(product([0, 1, -1], repeat=len(A)))
            #     # del possible_directions[0]  # remove 0-direction
            #     ord = np.argsort(np.abs(g))[::-1]

            #     v = np.zeros(len(g))
            #     v_best = v.copy()

            #     dir_deriv = 1e8
            #     idx_best = np.ones(len(g))

            #     split = []

            #     s = -np.sign(g)

            #     # search all directions for best direction
            #     for i in range(len(g)):
            #         v[ord[i]] = s[ord[i]]
            #         v /= norm(v)  # normalize direction

            #         # smallest epsilon such that current clustering is maintained
            #         idx = np.flip(np.argsort(np.abs(h0 * v)))
            #         sgn = np.sign(beta[C_j] + h0 * v)

            #         d = np.dot(v, g) + np.sum(lambdas_j[idx] * v * sgn)

            #         if d < dir_deriv:
            #             split.append(ord[i])
            #             dir_deriv = d
            #             v_best = v.copy()
            #             idx_best = idx.copy()

            #     if len(split) < len(C_j) and len(split) > 0:
            #         split_cluster(C, C_ptr, c, j, sorted(split))
            #         g = g[split]
            #         C_j = C[C_ptr[j] : C_ptr[j + 1]].copy()
            #         do_split = True

            # if len(c_ind_j) > 1 and epoch % split_freq == 0:
            #     # check if clusters should split and if so how
            #     x = beta[c_ind_j] - g / L
            #     x = -g
            #     split = find_splits(x, lambdas_j)
            #     lambda_sum = 0.0
            #     grad_sum = 0.0
            #     ord = np.argsort(np.abs(g))[::-1]

            #     k = 0

            #     # if abs(g[ord[0]]) > lambdas_j[0]:
            #     #     split = []
            #     #     for k, ord_k in enumerate(ord):
            #     #         grad_sum += abs(g[ord_k])
            #     #         lambda_sum += lambdas_j[k]
            #     #         if grad_sum > lambda_sum:
            #     #             split.append(ord_k)
            #     #         else:
            #     #             break

            #     #     split.sort()
            #     #     split_cluster(C, C_ptr, c, j, split)
            #     #     g = g[split]
            #     #     C_j = C[C_ptr[j] : C_ptr[j + 1]].copy()
            #     #     break

            #     if len(split) < len(c_ind_j):
            #         do_split = True
            #         # C = [C[i] for i in split]
            #         # clusters.split(j, split)
            #         # C_ptr, c = split_cluster(C, C_ptr, c, j, split)
            #         split_cluster(C, c_ptr, c, j, sorted(split))
            #         g = g[split]
            #         c_ind_j = C[c_ptr[j] : c_ptr[j + 1]].copy()

            s = np.sign(beta[c_ind_j]) if c[j] != 0 else -np.sign(g)
            # s = np.sign(beta[C_j]) if c[j] != 0 else np.ones(len(C_j))

            sum_X = X[:, c_ind_j] @ s
            L_j = sum_X.T @ sum_X / n
            x = c[j] - (s.T @ g) / L_j

            beta_tilde, ind_new = slope_threshold(
                x, lambdas / L_j, c_ind, c_ptr, c, n_c, j
            )

            beta[c_ind[c_ptr[j] : c_ptr[j + 1]]] = beta_tilde * s
            # update_cluster(C, C_ptr, c, j, c_new)

            c_old = c[j]

            if c_old != abs(beta_tilde):
                r -= (c[j] - abs(beta_tilde)) * sum_X

            # r -= (c[j] - beta_tilde) * sum_X
            coef_new = abs(beta_tilde)

            # if j == 1:
            #     raise ValueError()

            # C_true, C_ptr_true, c_true = initialize_clusters(beta)
            # C_prev = c_ind.copy()
            # C_ptr_prev = c_ptr.copy()
            # c_prev = c.copy()

            # if c[j] == c_new and do_split:
            #     # update did not change cluster, merge clusters back together
            #     if c[j] != c[j + 1]:
            #         raise ValueError("clusters are not the same")
            #     merge_clusters(c_ind, c_ptr, c, j + 1, j)

            if do_cluster_updates:
                ind_old = j
                n_c = update_cluster(c, c_ptr, c_ind, n_c, coef_new, ind_old, ind_new)
            else:
                c[j] = coef_new

            # if not (c_ind == C_true and c_ptr == C_ptr_true and c == c_true):
            #     print("C_ptr_upda", c_ptr)
            #     print("C_ptr_true", C_ptr_true)
            #     print("C_upda", c_ind)
            #     print("C_true", C_true)
            #     print("c_upda", c)
            #     print("c_true", c_true)
            #     raise ValueError("Cluster updating is not working")

            features_seen += len(c_ind_j)

            if cyclic:
                j += 1

    features_seen -= p  # keep approximate epoch consistency

    epoch += 1

# beta_star, primals_star, gaps_star, theta_star = prox_grad(
#     X, y, lambdas, max_epochs=1000, verbose=False
# )

plt.clf()
# plt.hlines(0, xmin=0, xmax=maxit, color="lightgrey")
plt.semilogy(np.arange(len(gaps)), gaps)
plt.vlines(np.arange(0, epoch, pgd_freq), np.min(gaps), np.max(gaps), linestyles="dotted")
plt.show(block=False)
