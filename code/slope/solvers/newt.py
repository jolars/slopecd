from timeit import default_timer as timer

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.random import default_rng
from scipy import sparse, stats
from scipy.linalg import inv, norm, solve
from scipy.optimize import minimize

from slope import sorted_l1_proj
from slope.solvers import prox_grad
from slope.utils import dual_norm_slope, prox_slope


def permutation_matrix(x):
    n = len(x)

    signs = np.sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = np.zeros((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[ord_j]

    return pi


def psi(y, x, A, b, lambdas, sigma):
    ATy = A.T @ y

    # TODO(jolars): not at all sure what epsilon and lambdas should be here
    # epsilon = np.sum(np.abs(ATy))
    w = x - sigma * ATy
    # u = sorted_l1_proj(x_tilde, lambdas / sigma, epsilon / sigma)
    u = (1 / sigma) * (w - prox_slope(w, lambdas * sigma))

    phi = 0.5 * norm(u - w / sigma) ** 2

    return 0.5 * norm(y) ** 2 + b @ y - (0.5 / sigma) * norm(x) ** 2 + sigma * phi


rng = default_rng(9)

m = 100
n = 500

A = rng.standard_normal((m, n))
b = rng.standard_normal(m)

# generate lambdas
randnorm = stats.norm(loc=0, scale=1)
q = 0.3
lambdas_seq = randnorm.ppf(1 - np.arange(1, n + 1) * q / (2 * n))
lambda_max = dual_norm_slope(A, b, lambdas_seq)

lambdas = lambda_max * lambdas_seq / 5

# step size
sigma = 1

# line search parameters
mu = 0.2
delta = 0.5
beta = 2

# CG parameters
eta = 0.5
tau = 0.5

# step 1 update parameters
epsilon_k = 1e-3
delta_k = 1e-3
delta_prime_k = 1e-3

m, n = A.shape

r = b.copy()
# x = np.zeros(n)
x = rng.standard_normal(n)
# x = np.array([3, 3, 0])
y = rng.standard_normal(m)
theta = np.zeros(m)

max_epochs = 100
max_inner_it = 1000
gap_freq = 1
max_time = np.inf
tol = 1e-8

primals, gaps = [], []
primals.append(norm(b) ** 2 / (2 * m))
gaps.append(primals[0])

time_start = timer()

verbose = False

B = np.eye(n) - np.eye(n, k=1)

x_diff_norm = 0

for epoch in range(max_epochs):
    # step 1
    delta_prime_k *= 0.5
    epsilon_k *= 0.9
    delta_k *= 0.9

    for j in range(max_inner_it):
        # step 1a, compute the newton direction
        x_tilde = x / sigma - (A.T @ y)

        # construct M
        pi = permutation_matrix(x_tilde)  # pi @ x == np.sort(np.abs(x))[::-1]

        # construct Jacobian
        # P = jacobian(pi @ w, pi, B, lambdas)
        x_lambda = pi @ prox_slope(x_tilde, sigma)
        z = solve(B @ B.T, B @ (pi @ x_tilde - lambdas - x_lambda))

        if verbose:
            print(f"pi: {pi}")
            print(f"x_tilde: {x_tilde}")
            print(f"x_lambda: {x_lambda}")
            print(f"z: {z}")

        z_supp = np.where(z != 0)[0]
        I_x_lambda = np.where(B @ x_lambda == 0)[0]

        Gamma = np.intersect1d(z_supp, I_x_lambda)

        B_Gamma = B[Gamma, :]

        P = np.eye(n) - B_Gamma.T @ solve(B_Gamma @ B_Gamma.T, B_Gamma)

        M = solve(pi, P) @ pi

        V = np.eye(m) + sigma * (A @ M @ A.T)

        nabla_psi = y + b - A @ prox_slope(x - sigma * (A.T @ y), sigma * lambdas)

        d = solve(V, -nabla_psi)

        if norm(V @ d + nabla_psi) > min(eta, norm(nabla_psi) ** (1 + tau)):
            raise ValueError("this should not happen")

        # check for convergence
        norm_nabla_psi = norm(nabla_psi)

        if norm_nabla_psi <= epsilon_k / np.sqrt(sigma) or (
            norm_nabla_psi <= (delta_k / np.sqrt(sigma)) * x_diff_norm
            and norm_nabla_psi <= (delta_prime_k / sigma) * x_diff_norm
        ):
            break

        if verbose:
            print("P\n", P)
            print("M\n", M)
            print("d\n", d)
            print("nabla_psi\n", nabla_psi)
            print("norm(nabla_psi)\n", norm(nabla_psi))

        # step 1b, line search
        mj = 0

        while True:
            alpha = delta**mj

            lhs = psi(y + alpha * d, x, A, b, lambdas, sigma)
            rhs = psi(y, x, A, b, lambdas, sigma) + mu * alpha * nabla_psi @ d

            if verbose:
                print(f"lhs: {lhs}, rhs: {rhs}, m_j: {mj}")

            if lhs <= rhs:
                break
            else:
                mj = 1 if mj == 0 else mj * beta

        # step 1c, update y
        y = y + alpha * d

    # step 2, update x
    x_old = x.copy()
    x = prox_slope(x - sigma * (A.T @ y), sigma * lambdas)
    x_diff_norm = norm(x - x_old)

    # step 3, update sigma
    # TODO(jolars): The paper says nothing about how sigma is updated except
    # that it is always increased if I interpret the paper correctly.
    # But in correspondence with the authors, they say that it is decreased or
    # increased based on the primal and dual residuals.
    sigma *= 1.1

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

        # if verbose:
        print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")

        if gap < tol or times_up:
            break

# w_pgd, primals_pgd, gaps_pgd, _ = prox_grad(A, b, lambdas / m, max_epochs=max_epochs)

# return x, primals, gaps, times
