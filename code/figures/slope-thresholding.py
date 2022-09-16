from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from slope.utils import dual_norm_slope


def get_clusters(w):
    unique, indices, counts = np.unique(
        np.abs(w), return_inverse=True, return_counts=True
    )

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

for a in a_list:
    k = np.where(np.logical_and(np.sign(a) * a <= l_sums, np.sign(a) * a >= r_sums))[0]

    if np.abs(a) < zerosum:
        res.extend([0.0])
    elif len(k) != 0:
        l_sum = l_sums[k][0]
        r_sum = r_sums[k][0]

        res.extend([np.sign(a) * c_wo_i[k]])
    else:
        ll = len(l_sums) - bisect_left(l_sums[::-1], np.abs(a))
        res.extend([np.sign(a) * (np.abs(a) - sums[ll])])

plt.rcParams["text.usetex"] = True

plt.close("all")

fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)

ax.hlines(0, xmin=min(a_list), xmax=max(a_list), color="lightgrey")

lambda_sums = np.sort(
    np.hstack((-l_sums, -r_sums, l_sums, r_sums, sums[-1], -sums[-1]))
)

ax.vlines(
    lambda_sums,
    ymin=min(res),
    ymax=max(res),
    linestyles="dotted",
)

xlim = ax.get_xlim()

xy = np.stack(
    (
        ((lambda_sums - xlim[0]) / (xlim[1] - xlim[0])),
        np.repeat([1.01], len(lambda_sums)),
    ),
    axis=1,
)

x2_labs = (
    r"\(-\omega c^{\setminus k}_1 - \sum_{j \in C(c^{\setminus k}_1 + \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_1 + \varepsilon_c}}\)",
    r"\(-\omega c^{\setminus k}_1 - \sum_{j \in C(c^{\setminus k}_1 - \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_1 - \varepsilon_c}}\)",
    r"\(-\omega c^{\setminus k}_2 - \sum_{j \in C(c^{\setminus k}_2 + \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_2 + \varepsilon_c}}\)",
    r"\(-\omega c^{\setminus k}_2 - \sum_{j \in C(c^{\setminus k}_2 - \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_2 - \varepsilon_c}}\)",
    r"-\(\sum_{j \in C(0)} \lambda_{(j)^-_{\varepsilon_c}}\)",
    r"\(\sum_{j \in C(0)} \lambda_{(j)^-_{\varepsilon_c}}\)",
    r"\(\omega c^{\setminus k}_2 + \sum_{j \in C(c^{\setminus k}_2 - \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_2 - \varepsilon_c}}\)",
    r"\(\omega c^{\setminus k}_2 + \sum_{j \in C(c^{\setminus k}_2 + \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_2 + \varepsilon_c}}\)",
    r"\(\omega c^{\setminus k}_1 + \sum_{j \in C(c^{\setminus k}_1 - \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_1 - \varepsilon_c}}\)",
    r"\(\omega c^{\setminus k}_1 + \sum_{j \in C(c^{\setminus k}_1 + \varepsilon_c)} \lambda_{(j)^-_{c^{\setminus k}_1 + \varepsilon_c}}\)",
)

ax.plot(a_list, res, "-", color="black")
ax.set_ylabel(r"\(T(\gamma, \omega; c, \lambda)\)")
ax.set_xlabel(r"\(\gamma\)")

ax2_x = ax.secondary_xaxis("top")
ax2_x.set_xticks(
    lambda_sums,
    labels=x2_labs,
    rotation=60,
    verticalalignment="bottom",
    horizontalalignment="left",
)

y2_labs = (
    r"$-c^{\setminus k}_1$",
    r"$-c^{\setminus k}_2$",
    r"$c^{\setminus k}_2$",
    r"$c^{\setminus k}_1$",
)

y2_vals = np.sort(np.hstack((-np.delete(c, 1), np.delete(c, 1))))

ax2_y = ax.secondary_yaxis("right")
ax2_y.set_yticks(y2_vals, y2_labs)

# plt.show(block=False)

plt.rcParams["text.usetex"] = True
plt.savefig("../figures/slope-thresholding.pdf", bbox_inches="tight", pad_inches=0.01)
