import warnings
from timeit import default_timer as timer

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.random import default_rng
from scipy import sparse, stats
from scipy.linalg import cho_factor, cho_solve, inv, norm, solve
from scipy.optimize import minimize
from scipy.sparse.linalg import cg, spsolve

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


def build_W(x_tilde, sigma, lambdas, A):
    m = A.shape[0]

    ord = np.argsort(np.abs(x_tilde))[::-1]
    x_lambda = np.abs(prox_slope(x_tilde, sigma)[ord])

    z = slopep.BBT_inv_B(np.abs(x_tilde[ord]) - lambdas - x_lambda)

    z_supp = np.where(z != 0)[0]
    I_x_lambda = np.where(slopep.B(x_lambda) == 0)[0]

    Gamma = np.intersect1d(z_supp, I_x_lambda)
    GammaC = np.setdiff1d(np.arange(x_tilde.shape[0]), Gamma)

    start = 0
    nC = GammaC.shape[0]

    if sparse.issparse(A):
        VW = sparse.lil_matrix((m, nC), dtype=float)
    else:
        VW = np.zeros((m, nC))

    pi_list, piT_list = slopep.build_pi(x_tilde)

    for i in range(nC):
        ind = np.arange(start, GammaC[i] + 1)
        for j in ind:
            VW[:, i] += pi_list[j, 1] * A[:, pi_list[j, 0]]
        if ind.shape[0] > 1:
            VW[:, i] /= np.sqrt(ind.shape[0])
        start = GammaC[i] + 1

    if sparse.issparse(A):
        # NOTE: Use CSR here because transpose later on causes it to become CSC,
        # which is what we really want.
        VW = sparse.csr_matrix(VW)

    return np.sqrt(sigma) * VW


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


def compute_direction(x, sigma, A, b, y, ATy, lambdas, cg_param, solver):

    x_tilde = x / sigma - ATy

    nabla_psi = y + b - A @ prox_slope(x - sigma * ATy, sigma * lambdas)

    W = build_W(x_tilde, sigma, lambdas, A)

    m, r1_plus_r2 = W.shape

    if solver == "auto":
        if r1_plus_r2 <= 100 * m:
            solver = "woodbury"
        elif m > 10_000 and m / r1_plus_r2 > 0.1:
            solver = "cg"
        else:
            solver = "standrad"

    if solver == "woodbury":
        # Use Woodbury factorization solve
        WTW = W.T @ W
        if sparse.issparse(A):
            V_inv = sparse.eye(m, format="csc") - W @ spsolve(
                sparse.eye(r1_plus_r2, format="csc") + WTW, W.T
            )
        else:
            np.fill_diagonal(WTW, WTW.diagonal() + 1)
            V_inv = -(W @ solve(WTW, W.T))
            np.fill_diagonal(V_inv, V_inv.diagonal() + 1)

        d = V_inv @ (-nabla_psi)
    elif solver == "cg":
        # Use conjugate gradient
        eta = cg_param["eta"]
        tau = cg_param["tau"]
        tol = cg_param["starting_tol"]
        d = np.zeros(m)

        while True:
            V = W @ W.T
            if sparse.issparse(A):
                V += sparse.eye(m, format="csc")
            else:
                np.fill_diagonal(V, V.diagonal() + 1)

            M = sparse.diags(V.diagonal())  # preconditioner
            d, _ = cg(V, -nabla_psi, tol=tol, M=M, x0=d)

            if norm(V @ d + nabla_psi) <= min(eta, norm(nabla_psi) ** (1 + tau)):
                break
            else:
                tol *= 0.1
    else:
        V = W @ W.T
        if sparse.issparse(A):
            V = sparse.eye(V.shape[0]) + W @ W.T
            d = spsolve(V, -nabla_psi)
        else:
            np.fill_diagonal(V, V.diagonal() + 1)
            d = cho_solve(cho_factor(V), -nabla_psi)

    return d, nabla_psi


def line_search(y, d, x, A, ATy, ATd, b, lambdas, sigma, nabla_psi, line_search_param):
    # step 1b, line search
    mj = 0

    psi0 = psi2(y, x, ATy, b, lambdas, sigma)
    nabla_psi_d = nabla_psi @ d
    while True:
        alpha = line_search_param["delta"] ** mj
        lhs = psi2(y + alpha * d, x, ATy + alpha * ATd, b, lambdas, sigma)
        rhs = psi0 + line_search_param["mu"] * alpha * nabla_psi_d

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

    return crit_A and crit_B1 and crit_B2


def inner_step(
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
    solver,
):

    sigma = local_param["sigma"]
    d, nabla_psi = compute_direction(
        x_old, sigma, A, b, y, ATy, lambdas, cg_param, solver
    )
    ATd = A.T @ d
    alpha = line_search(
        y, d, x_old, A, ATy, ATd, b, lambdas, sigma, nabla_psi, line_search_param
    )

    # step 1c, update y
    y += alpha * d
    ATy += alpha * ATd

    # step 2, update x
    x = prox_slope(x_old - sigma * ATy, sigma * lambdas)

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


def newt_alm(
    A,
    b,
    lambdas,
    max_epochs=1000,
    tol=1e-6,
    max_time=np.inf,
    gap_freq=1,
    max_inner_it=1_000,
    solver="auto",
    line_search_param={"mu": 0.2, "delta": 0.5, "beta": 2},
    cg_param={"eta": 1e-4, "tau": 0.1, "starting_tol": 1e-1},
    verbose=True,
):
    if solver not in ["auto", "standard", "woodbury", "cg"]:
        raise ValueError("`solver` must be one of auto, standard, woodbury, and cg")

    m, n = A.shape

    lambdas *= m

    # step 1 update parameters
    local_param = {"epsilon": 1, "delta": 1, "delta_prime": 1, "sigma": 0.5}

    x = np.zeros(n)
    y = np.zeros(m)
    r = b.copy()
    theta = np.zeros(m)
    primals, gaps = [], []
    primals.append(norm(b) ** 2 / (2 * m))
    gaps.append(primals[0])

    times = []
    time_start = timer()
    times.append(timer() - time_start)

    ATy = A.T @ y
    for epoch in range(max_epochs):
        # step 1
        local_param["delta_prime"] *= 0.999
        local_param["epsilon"] *= 0.9
        local_param["delta"] *= 0.999

        x_old = x.copy()

        for j in range(max_inner_it):
            converged, x, y, ATy = inner_step(
                A,
                b,
                x_old,
                y,
                ATy,
                lambdas,
                x_old,
                local_param,
                line_search_param,
                cg_param,
                solver,
            )
            if converged:
                break

            if j == max_inner_it - 1:
                warnings.warn("The inner solver did not converge.")

        # step 3, update sigma
        # TODO(jolars): The paper says nothing about how sigma is updated except
        # that it is always increased if I interpret the paper correctly.
        # But in correspondence with the authors, they say that it is decreased or
        # increased based on the primal and dual residuals.
        local_param["sigma"] *= 1.1

        times_up = timer() - time_start > max_time

        if epoch % gap_freq == 0 or times_up:
            r[:] = b - A @ x
            theta = r / m
            theta /= max(1, dual_norm_slope(A, theta, lambdas / m))

            primal = (0.5 / m) * norm(r) ** 2 + np.sum(
                lambdas * np.sort(np.abs(x))[::-1]
            ) / m
            dual = (0.5 / m) * (norm(b) ** 2 - norm(b - theta * m) ** 2)

            primals.append(primal)
            gap = (primal - dual) / max(1, primal)
            gaps.append(gap)
            times.append(timer() - time_start)

            if verbose:
                print(f"Epoch: {epoch + 1}, loss: {primal}, gap: {gap:.2e}")

            if gap < tol:
                break

    return x, primals, gaps, times
