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

s = np.sign(beta)

n_vals = 100
c_vals = np.linspace(-0.8, 0.8, num=n_vals)
c_vals = np.sort(np.concatenate((c, -c, [0], c_vals)))

cs = []
grad = []
obj = []
obj_cs = []

i = 1

for k, z in enumerate(c_vals):
    z_in_c = abs(z) == abs(c)
    z_in_c[i] = False

    Ci = C[i]

    z_sign = np.sign(z)
    c[i] = np.abs(z)

    ord = np.argsort(c)[::-1]

    if any(z_in_c):
        j = np.where(z_in_c)[0][0]
        if ord[i] > ord[j]:
            if z_sign == -1:
                ord[i], ord[j] = ord[j], ord[i]

        if ord[i] < ord[j]:
            if z_sign == 1:
                ord[i], ord[j] = ord[j], ord[i]

    C = C[ord]
    c = c[ord]
    C_size = C_size[ord]
    beta[Ci] = s[Ci] * z

    # new index for cluster
    i = ord[i]

    z_sign = np.sign(z)

    A = np.zeros(len(beta), dtype=bool)

    A[Ci] = True
    B = ~A

    csum = np.concatenate(([0], np.cumsum(C_size)))

    lam = lambdas[range(csum[i], csum[i + 1])]

    g = -y.T @ X[:, A] @ s[A] + (X[:, B] @ beta[B]).T @ X[:, A] @ s[A] + \
        z * norm(X[:, A] @ s[A])**2

    tmp = 0.5 * norm(y - X[:, B] @ beta[B] - z * X[:, A] @ s[A])**2 + \
        np.sum(np.sort(np.abs(beta))[::-1] * lambdas)
    obj.append(tmp)
    obj_cs.append(z)

    if z == 0:
        grad.append(g - np.sum(lam))
        grad.append(g + np.sum(lam))
        cs.append(z)
        cs.append(z)
    elif any(z_in_c):
        if z_sign == 1:
            next_lam = lambdas[range(csum[i] - 1, csum[i + 1] - 1)]
        else:
            next_lam = lambdas[range(csum[i] + 1, csum[i + 1] + 1)]

        grad.append(g + z_sign * np.sum(lam))
        grad.append(g + z_sign * np.sum(next_lam))
        cs.append(z)
        cs.append(z)
    else:
        grad.append(g + z_sign * np.sum(lam))
        cs.append(z)

# plt.clf()

plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(6.1, 3.1))

c_out = np.delete(c, i)
tmp = np.hstack((-c_out, 0, c_out))

axs[0].vlines(tmp,
              ymin=min(grad),
              ymax=max(grad),
              linestyles="dotted",
              color="black")
axs[0].hlines(0, xmin=min(cs), xmax=max(cs), color="lightgrey")
axs[0].plot(cs, grad, color="black")
axs[0].set_ylabel("$\\frac{\\partial}{\\partial_z} P(\\beta)$")
axs[0].set_xlabel("$z$")

axs[1].vlines(tmp,
              ymin=min(obj),
              ymax=max(obj),
              linestyles="dotted",
              color="black")
axs[1].plot(obj_cs, obj, color="black")
axs[1].set_ylabel("$P(\\beta)$")
axs[1].set_xlabel("$z$")

plt.tight_layout()
plt.savefig("../figures/clusterupdate-grad-obj.pdf",
            bbox_inches="tight",
            pad_inches=0.01)
