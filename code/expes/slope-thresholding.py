from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import stats

from slope.utils import dual_norm_slope


def get_clusters(w):
    unique, indices, counts = np.unique(np.abs(w),
                                        return_inverse=True,
                                        return_counts=True)

    clusters = [[] for _ in range(len(unique))]
    for i in range(len(indices)):
        clusters[indices[i]].append(i)
    return clusters[::-1], counts[::-1], indices[::-1], unique[::-1]


np.random.seed(10)

n = 10
p = 4

X = np.random.rand(n, p)
beta = np.random.rand(p)
y = X @ beta + np.random.rand(n)

randnorm = stats.norm(loc=0, scale=1)
q = 0.8

lambdas = randnorm.ppf(1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))
lambda_max = dual_norm_slope(X, y / len(y), lambdas)
lambdas = lambda_max * lambdas * 0.5

beta = np.array([0.5, -0.5, 0.3, 0.7])

C, C_size, c_indices, c = get_clusters(beta)
C = np.array(C)
C_size = np.array(C_size)
C_start = np.concatenate(([0], np.cumsum(list(map(len, np.delete(C, -1))))))
C_end = C_start + C_size

s = np.sign(beta)

n_vals = 100
c_vals = np.linspace(-0.8, 0.8, num=n_vals)
c_vals = np.sort(np.concatenate((c, -c, [0], c_vals)))

i = 1

l_sums = []
r_sums = []
sums = []

for j in range(len(C)):
    if j == i:
        continue

    mod = len(C[i]) if j > i else 0

    # check upper end of cluster
    l_start = C_start[j] - mod
    l_end = C_start[j] + len(C[i]) - mod

    # check lower end of cluster
    r_start = C_end[j] - mod
    r_end = C_end[j] + len(C[i]) - mod

    l_sum = sum(lambdas[l_start:l_end])
    r_sum = sum(lambdas[r_start:r_end])

    l_sums.extend([l_sum + c[j]])
    r_sums.extend([r_sum + c[j]])

    sums.extend([l_sum, r_sum])

l_sums = np.array(l_sums)
r_sums = np.array(r_sums)

sums = np.sort(np.unique(sums))[::-1]

c_wo_i = np.delete(c, i)
a_list = np.sort(np.hstack((sums, l_sums, r_sums, np.linspace(0, 2, 100))))
a_list = np.sort(np.hstack((-a_list, a_list)))
res = []

zerosum = np.sum(lambdas[::-1][range(2)])
lastsum = np.sum(lambdas[range(2)])

for a in a_list:
    k = np.where(
        np.logical_and(np.sign(a) * a <= l_sums,
                       np.sign(a) * a >= r_sums))[0]

    if np.abs(a) < zerosum:
        res.extend([0.0])
    elif len(k) != 0:
        l_sum = l_sums[k][0]
        r_sum = r_sums[k][0]

        res.extend([np.sign(a) * c_wo_i[k]])
    else:
        l = len(l_sums) - bisect_left(l_sums[::-1], np.abs(a))
        res.extend([np.sign(a) * (np.abs(a) - sums[l])])

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(4.2,2.8))
ax.hlines(0, xmin=min(a_list), xmax=max(a_list), color="lightgrey")
ax.vlines(np.hstack((-l_sums, -r_sums, l_sums, r_sums, sums[-1], -sums[-1])),
          ymin=min(res),
          ymax=max(res),
          linestyles="dotted",
          color="black")
ax.plot(a_list, res, '-', color="black")
ax.set_ylabel("$T(v,\\lambda)$")
ax.set_xlabel("$v$")
# plt.show(block=False)

plt.tight_layout()
plt.savefig("../figures/slope-thresholding.pdf",
            bbox_inches="tight",
            pad_inches=0.01)
