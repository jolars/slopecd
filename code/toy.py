from scipy import stats
from numba import njit
import numpy as np
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
import matplotlib.pyplot as plt

from slope.utils import ST, dual_norm_slope, prox_slope


@njit
def do_cd_epochs(n_cd, w, X, R, alphas, lc):
    n_samples = X.shape[0]
    for _ in range(n_cd):
        # do CD epochs pretending coefs order is fixed
        order = np.argsort(np.abs(w))[::-1]
        for idx, j in enumerate(order):  # update from big to small
            old = w[j]
            w[j] = ST(w[j] + X[:, j] @ R / (lc[j] * n_samples),
                      alphas[idx] / lc[j])
            if w[j] != old:
                R += (old - w[j]) * X[:, j]


def prox_grad(X, y, alphas, max_iter=100, tol=1e-10, n_cd=0, verbose=True):
    n_samples, n_features = X.shape
    R = y.copy()
    w = np.zeros(n_features)

    L = norm(X, ord=2) ** 2 / n_samples
    lc = norm(X, axis=0) ** 2 / n_samples
    E, gaps = [], []
    E.append(norm(y)**2 / (2 * n_samples))
    gaps.append(E[0])
    for t in range(max_iter):
        w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
        R[:] = y - X @ w

        do_cd_epochs(n_cd, w, X, R, alphas, lc)

        theta = R / n_samples
        theta /= max(1, dual_norm_slope(X, theta, alphas))
        dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
            (2 * n_samples)
        primal = norm(R) ** 2 / (2 * n_samples) + \
            np.sum(alphas * np.sort(np.abs(w))[::-1])
        E.append(primal)
        gap = primal - dual
        gaps.append(gap)
        if verbose:
            print(f"Iter: {t + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol:
            break
    return w, E, gaps, theta


X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))

# alphas_seq = np.ones(X.shape[1])

alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 500
plt.close('all')

for n_cd in [0, 1, 5, 10]:
    w, E, gaps, theta = prox_grad(
        X, y, alphas, max_iter=1000 // (n_cd + 1), n_cd=n_cd, verbose=1)
    print(gaps[0])

    plt.semilogy(np.arange(len(E)) * (1 + n_cd), gaps,
                 label=f'n_cd = {n_cd}')
plt.legend()
plt.show(block=False)
