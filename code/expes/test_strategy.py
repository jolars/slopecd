import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data
from numpy.linalg import norm
from slope.utils import dual_norm_slope, slope_threshold
from slope.utils import get_clusters, prox_slope
from slope.solvers import prox_grad, oracle_cd

X, y, _ = make_correlated_data(n_samples=100, n_features=40, random_state=0)
randnorm = stats.norm(loc=0, scale=1)
q = 0.5

alphas_seq = randnorm.ppf(
    1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))


alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

alphas = alpha_max * alphas_seq / 100
plt.close('all')

max_iter = 1000
tol = 1e-12

n_samples, n_features = X.shape
R = y.copy()
w = np.zeros(n_features)
theta = np.zeros(n_samples)

L = norm(X, ord=2)**2 / n_samples
lc = norm(X, axis=0)**2 / n_samples
E, gaps = [], []
E.append(norm(y)**2 / (2 * n_samples))
gaps.append(E[0])

for t in range(max_iter):

    if t % 5 == 0:
        w = prox_slope(w + (X.T @ R) / (L * n_samples), alphas / L)
        R[:] = y - X @ w
        g = X.T @ R
        theta = R / n_samples
        theta /= max(1, dual_norm_slope(X, theta, alphas))
        dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
            (2 * n_samples)
        primal = norm(R) ** 2 / (2 * n_samples) + \
            np.sum(alphas * np.sort(np.abs(w))[::-1])

        E.append(primal)
        gap = primal - dual
        gaps.append(gap)
    else:
        C, C_size, C_indices, c = get_clusters(w)
        C_end = np.cumsum(C_size)
        C_start = C_end - C_size
        C_ord = np.arange(len(C))
        for j in range(len(C)):
            A = C[j].copy()
            grad_A = X[:, A].T @ R
            c_old = c[j]
            B = list(set(range(n_features)) - set(A))
            # s = np.sign(beta[A])
            s = np.sign(w[A])
            s = np.ones(len(s)) if all(s == 0) else s
            L_j = s.T @ X[:, A].T @ X[:, A] @ s / n_samples

            x = np.abs(w[A][0]) + (s.T @ g[A]) / (L_j * n_samples)
            # beta_tilde = ST
            beta_tilde = slope_threshold(
                x, alphas/L_j, C, C_start, C_end, c, j)
            c[j] = np.abs(beta_tilde)
            w[A] = beta_tilde * s
            R[:] = y - X @ w
            g = X.T @ R

        theta = R / n_samples
        theta /= max(1, dual_norm_slope(X, theta, alphas))
        dual = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
            (2 * n_samples)
        primal = norm(R) ** 2 / (2 * n_samples) + \
            np.sum(alphas * np.sort(np.abs(w))[::-1])

        E.append(primal)
        gap = primal - dual
        gaps.append(gap)

    print(f"Iter: {t + 1}, loss: {primal}, gap: {gap:.2e}")
    if gap < tol:
        break


beta_star, primals_star, gaps_star, theta_star = prox_grad(
    X, y, alphas, max_iter=1000, n_cd=0, verbose=True, tol=tol,
)
beta_oracle, primals_oracle, gaps_oracle = oracle_cd(
    X, y, alphas, max_iter=1000, verbose=True, tol=tol,
)
plt.semilogy(np.arange(len(gaps)), gaps, label='cd')
plt.semilogy(np.arange(len(gaps_star)), gaps_star, label='pgd')
plt.semilogy(np.arange(len(gaps_oracle)), gaps_oracle, label='oracle')

plt.legend()
plt.show()
