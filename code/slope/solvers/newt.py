from timeit import default_timer as timer

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.random import default_rng
from scipy import sparse, stats
from scipy.linalg import inv, norm, solve
from scipy.optimize import minimize

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope, prox_slope


def permutation_matrix(x):
    n = len(x)

    signs = np.sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = np.zeros((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[j]

    return pi


def jacobian(w, pi, B, lambdas):
    n = len(w)

    x = pi @ prox_slope(w, lambdas)
    z = solve(B @ B.T, B @ (w - lambdas - x))

    # z_supp = np.where(z != 0)[0]
    # Bx_supp = np.where(B @ x == 0)

    # supp = np.where(B @ x == 0)[0]
    supp = np.where(z != 0)[0]

    B_s = B[supp, :]

    P = np.eye(n) - B_s.T @ solve(B_s @ B_s.T, B_s)

    return P


# def sl1_norm_conjugate():

# minimize_u (1 / sigma) * kappa_star(u) + 0.5 * ||u-x||^2
# def find_phi(x, b):


def L(y, x, A, b, sigma, phi):
    return 0.5 * norm(b) ** 2 + b @ y - (0.5 / sigma) * norm(x) ** 2 + sigma * phi


# def newt_alm(
#     A,
#     b,
#     lambdas,
#     max_epochs=100,
#     tol=1e-10,
#     max_time=np.inf,
#     gap_freq=1,
#     verbose=True,
# ):
# parameters

rng = default_rng(9)

m = 100
n = 3

A = rng.standard_normal((m, n))
b = rng.standard_normal(m)

# generate lambdas
randnorm = stats.norm(loc=0, scale=1)
q = 0.2
lambdas_seq = randnorm.ppf(1 - np.arange(1, n + 1) * q / (2 * n))
lambda_max = dual_norm_slope(A, b, lambdas_seq)

lambdas = lambda_max * lambdas_seq / 100

sigma = 1
mu = 0.25
eta = 0.5
delta = 0.5
tau = 0.5
alpha = 0.01

m, n = A.shape

r = b.copy()
# x = np.zeros(n)
x = rng.standard_normal(n)
y = rng.standard_normal(m)
theta = np.zeros(m)

max_epochs = 10
gap_freq = 1
max_time = np.inf
verbose = True
tol = 1e-8

primals, gaps = [], []
primals.append(norm(b) ** 2 / (2 * m))
gaps.append(primals[0])

time_start = timer()

B = np.eye(n) - np.eye(n, k=1)

for epoch in range(max_epochs):
    # step 1
    for j in range(100):
        # step 1a, compute the newton direction
        w = x / sigma - (A.T @ y)

        # construct M
        pi = permutation_matrix(w)  # pi @ x == np.sort(np.abs(x))[::-1]
        P = jacobian(pi @ w, pi, B, lambdas)
        M = solve(pi, P) @ P

        V = np.eye(m) + sigma * (A @ M @ A.T)

        nabla_psi = y + b - A @ prox_slope(x - sigma * (A.T @ y), sigma * lambdas)

        d = solve(V, -nabla_psi)

        # step 1b, line search
        # mm = 0

        # while (
        #     psi(y + (delta**mm) * d) <= psi(y) + mu * (delta**mm) * nabla_psi(y) @ d
        # ):
        #     mm += 1

        # step 1c, update y
        y = y + alpha * d

    # step 2, update x
    x = prox_slope(x - sigma * (A.T @ y), sigma * lambdas)

    # step 3, update sigma
    # TODO: it's not actually clear from the paper how to do this
    sigma *= 2

    times_up = timer() - time_start > max_time

    if epoch % gap_freq == 0 or times_up:
        r[:] = b - A @ x
        theta = r / m
        theta /= max(1, dual_norm_slope(A, theta, lambdas / m))

        primal = (0.5 / m) * norm(r) ** 2 + np.sum(
            (lambdas / m) * np.sort(np.abs(x))[::-1]
        )
        dual = (0.5 / m) * (norm(b) ** 2 - norm(b - theta * m) ** 2)

        primals.append(primal)
        gap = primal - dual
        gaps.append(gap)
        # times.append(timer() - time_start)

        if verbose:
            print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")
        if gap < tol or times_up:
            break

w_pgd, primals_pgd, gaps_pgd, _ = prox_grad(A, b, lambdas / m, max_epochs=max_epochs)

# return x, primals, gaps, times
