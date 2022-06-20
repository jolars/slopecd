import warnings
from timeit import default_timer as timer

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.random import default_rng
from scipy import sparse, stats
from scipy.linalg import inv, norm, solve
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

import slope.permutation as slopep
from slope.clusters import get_clusters
from slope.solvers import prox_grad
from slope.utils import dual_norm_slope, prox_slope


# build the approximate Hessian
def build_AMAT_old(x_tilde, sigma, lambdas, A):

    B = sparse.eye(x_tilde.shape[0], format="csc") - sparse.eye(
        x_tilde.shape[0], k=1, format="csc"
    )
    pi = slopep.permutation_matrix(x_tilde)  # pi @ x == np.sort(np.abs(x))[::-1]
    ord = np.argsort(np.abs(x_tilde))[::-1]
    x_lambda = np.abs(prox_slope(x_tilde, sigma)[ord])

    z = slopep.BBT_inv_B(np.abs(x_tilde[ord]) - lambdas - x_lambda)

    z_supp = np.where(z != 0)[0]
    I_x_lambda = np.where(B @ x_lambda == 0)[0]

    Gamma = np.intersect1d(z_supp, I_x_lambda)

    B_Gamma = B[Gamma, :]

    P = sparse.eye(n, format="csc") - B_Gamma.T @ spsolve(B_Gamma @ B_Gamma.T, B_Gamma)

    M = pi.T @ P @ pi

    V = sigma * (A @ M @ A.T)

    return V


def build_AMAT(x_tilde, sigma, lambdas, A):

    ord = np.argsort(np.abs(x_tilde))[::-1]
    x_lambda = np.abs(prox_slope(x_tilde, sigma)[ord])

    z = slopep.BBT_inv_B(np.abs(x_tilde[ord]) - lambdas - x_lambda)

    z_supp = np.where(z != 0)[0]
    I_x_lambda = np.where(slopep.B(x_lambda) == 0)[0]

    Gamma = np.intersect1d(z_supp, I_x_lambda)
    GammaC = np.setdiff1d(np.arange(x_tilde.shape[0]), Gamma)

    start = 0
    nC = GammaC.shape[0]
    VW = np.zeros((m, nC))
    start = 0
    nC = GammaC.shape[0]
    VW = np.zeros((m, nC))
    pi_list, piT_list = slopep.build_pi(x_tilde)
    for i in range(nC):
        ind = np.arange(start, GammaC[i] + 1)
        for j in ind:
            VW[:, i] += pi_list[j, 1] * A[:, pi_list[j, 0]]
        if ind.shape[0] > 1:
            VW[:, i] /= np.sqrt(ind.shape[0])
        start = GammaC[i] + 1

    VW *= np.sqrt(sigma)
    return VW @ VW.T


def psi(y, x, A, b, lambdas, sigma):
    ATy = A.T @ y

    # TODO(jolars): not at all sure what epsilon and lambdas should be here
    # epsilon = np.sum(np.abs(ATy))
    w = x - sigma * ATy
    # u = sorted_l1_proj(x_tilde, lambdas / sigma, epsilon / sigma)
    u = (1 / sigma) * (w - prox_slope(w, lambdas * sigma))

    phi = 0.5 * norm(u - w / sigma) ** 2

    return 0.5 * norm(y) ** 2 + b @ y - (0.5 / sigma) * norm(x) ** 2 + sigma * phi


def psi2(y, x, ATy, b, lambdas, sigma):

    # TODO(jolars): not at all sure what epsilon and lambdas should be here
    # epsilon = np.sum(np.abs(ATy))
    w = x - sigma * ATy
    # u = sorted_l1_proj(x_tilde, lambdas / sigma, epsilon / sigma)
    u = (1 / sigma) * (w - prox_slope(w, lambdas * sigma))

    phi = 0.5 * norm(u - w / sigma) ** 2

    return 0.5 * norm(y) ** 2 + b @ y - (0.5 / sigma) * norm(x) ** 2 + sigma * phi


def compute_direction(x, sigma, A, y, ATy, lambdas, cg_param):

    x_tilde = x / sigma - ATy

    V = build_AMAT(x_tilde, sigma, lambdas, A)

    np.fill_diagonal(V, V.diagonal() + 1.0)

    nabla_psi = y + b - A @ prox_slope(x - sigma * ATy, sigma * lambdas)

    d = solve(V, -nabla_psi)

    if norm(V @ d + nabla_psi) > min(
        cg_param["eta"], norm(nabla_psi) ** (1 + cg_param["tau"])
    ):
        warnings.warn("Solver did not work")

    return d, nabla_psi


def line_search(y, d, x, A, ATy, ATd, b, lambdas, sigma, nabla_psi, line_search_param):
    # step 1b, line search
    mj = 0

    psi0 = psi2(y, x, ATy, b, lambdas, sigma)
    while True:
        alpha = line_search_param["delta"] ** mj

        lhs = psi2(y + alpha * d, x, ATy + alpha * ATd, b, lambdas, sigma)
        rhs = psi0 + line_search_param["mu"] * alpha * nabla_psi @ d

        if lhs <= rhs:
            break

        mj = 1 if mj == 0 else mj * line_search_param["beta"]

    return alpha


def check_convegence(x_diff_norm, nabla_psi, epsilon_k, sigma, delta_k, delta_prime_k):

    # check for convergence
    norm_nabla_psi = norm(nabla_psi)

    crit_A = norm_nabla_psi <= epsilon_k / np.sqrt(sigma)
    crit_B1 = norm_nabla_psi <= (delta_k / np.sqrt(sigma)) * x_diff_norm
    crit_B2 = norm_nabla_psi <= (delta_prime_k / sigma) * x_diff_norm

    if crit_A and crit_B1 and crit_B2:
        return True

    return False


def inner_step(
    A, b, x, y, ATy, lambdas, x_old, local_param, line_search_param, cg_param
):

    sigma = local_param["sigma"]
    d, nabla_psi = compute_direction(x, sigma, A, y, ATy, lambdas, cg_param)
    ATd = A.T @ d
    alpha = line_search(
        y, d, x, A, ATy, ATd, b, lambdas, sigma, nabla_psi, line_search_param
    )

    # step 1c, update y
    y += alpha * d
    ATy += alpha * ATd

    # step 2, update x
    x = prox_slope(x - sigma * ATy, sigma * lambdas)

    # check for convergence
    x_diff_norm = norm(x - x_old)

    converged = check_convegence(
        x_diff_norm,
        nabla_psi,
        local_param["epsilon"],
        sigma,
        local_param["delta"],
        local_param["delta_prime"],
    )
    return converged, x, y, ATy


def newton_solver(
    A,
    b,
    lambdas,
    x=None,
    y=None,
    optim_param={"max_epochs": 100, "max_inner_it": 10000, "tol": 1e-8, "gap_freq": 1},
    line_search_param={"mu": 0.2, "delta": 0.5, "beta": 2},
    cg_param={"eta": 1e-4, "tau": 0.5},
    verbose=True,
):

    m, n = A.shape

    if x is None:
        x = rng.standard_normal(n)
    if y is None:
        y = rng.standard_normal(m)
    max_epochs = optim_param["max_epochs"]
    max_inner_it = optim_param["max_inner_it"]

    # step 1 update parameters
    local_param = {"epsilon": 0.1, "delta": 0.1, "delta_prime": 0.1, "sigma": 1}

    r = b.copy()
    theta = np.zeros(m)
    primals, gaps = [], []
    primals.append(norm(b) ** 2 / (2 * m))
    gaps.append(primals[0])

    ATy = A.T @ y
    for epoch in range(max_epochs):
        # step 1
        local_param["delta_prime"] *= 0.9
        local_param["epsilon"] *= 0.9
        local_param["delta"] *= 0.9

        x_old = x.copy()

        for j in range(max_inner_it):

            converged, x, y, ATy = inner_step(
                A,
                b,
                x,
                y,
                ATy,
                lambdas,
                x_old,
                local_param,
                line_search_param,
                cg_param,
            )

            if converged:
                break

            if j == max_inner_it - 1:
                warnings.warn("The inner solver did not converge.")
                raise ValueError

        # step 3, update sigma
        # TODO(jolars): The paper says nothing about how sigma is updated except
        # that it is always increased if I interpret the paper correctly.
        # But in correspondence with the authors, they say that it is decreased or
        # increased based on the primal and dual residuals.
        local_param["sigma"] *= 1.1

        if epoch % optim_param["gap_freq"] == 0:
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

            if gap < optim_param["tol"]:
                break

    return x, gaps, primals


def problem1():
    rng = default_rng(9)

    m = 100
    n = 10

    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)

    # generate lambdas
    randnorm = stats.norm(loc=0, scale=1)
    q = 0.3
    lambdas_seq = randnorm.ppf(1 - np.arange(1, n + 1) * q / (2 * n))
    lambda_max = dual_norm_slope(A, b, lambdas_seq)

    lambdas = lambda_max * lambdas_seq / 5

    x = newton_solver(A, b, lambdas)

    return x


if __name__ == "__main__":
    rng = default_rng(9)

    m = 100
    n = 10

    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)

    # generate lambdas
    randnorm = stats.norm(loc=0, scale=1)
    q = 0.3
    lambdas_seq = randnorm.ppf(1 - np.arange(1, n + 1) * q / (2 * n))
    lambda_max = dual_norm_slope(A, b, lambdas_seq)

    lambdas = lambda_max * lambdas_seq / 5

    x_diff_norm = 0

    x = newton_solver(A, b, lambdas)
