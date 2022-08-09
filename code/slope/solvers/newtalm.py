import warnings
from timeit import default_timer as timer

import numpy as np
from numba import njit
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, norm, solve
from scipy.sparse.linalg import cg, spsolve

import slope.permutation as slopep
from slope.utils import add_intercept_column, dual_norm_slope, prox_slope2


def build_W(x_tilde, sigma, lambdas, A, fit_intercept):
    m = A.shape[0]

    ord = np.argsort(np.abs(x_tilde[fit_intercept:]))[::-1]
    x_lambda = np.abs(prox_slope2(x_tilde[fit_intercept:], lambdas)[ord])

    z = slopep.BBT_inv_B(np.abs(x_tilde[fit_intercept:][ord]) - lambdas - x_lambda)

    Gamma = np.where(np.logical_and(z != 0, slopep.B(x_lambda) == 0))[0]
    GammaC = np.setdiff1d(np.arange(len(x_lambda)), Gamma)

    nC = len(GammaC)

    pi_list, _ = slopep.build_pi(x_tilde[fit_intercept:])

    if sparse.issparse(A):
        W_row, W_col, W_data = slopep.assemble_sparse_W(
            nC, GammaC, pi_list, A.data, A.indices, A.indptr, m, fit_intercept
        )
        # we use CSR here because transpose later on makes it CSC, which
        # is what we really want.
        W = sparse.coo_matrix(
            (W_data, (W_row, W_col)), shape=(m, nC + fit_intercept)
        ).tocsr()
    else:
        W = slopep.assemble_dense_W(nC, GammaC, pi_list, A, fit_intercept)

    return np.sqrt(sigma) * W


@njit
def psi(y, x, ATy, b, lambdas, sigma, fit_intercept):
    w = x - sigma * ATy
    u = np.zeros(len(w))
    u[fit_intercept:] = (1 / sigma) * (
        w[fit_intercept:] - prox_slope2(w[fit_intercept:], lambdas * sigma)
    )
    if fit_intercept:
        u[0] = 0.0

    phi = 0.5 * np.linalg.norm(u - w / sigma) ** 2

    return (
        0.5 * np.linalg.norm(y) ** 2
        + b @ y
        - (0.5 / sigma) * np.linalg.norm(x) ** 2
        + sigma * phi
    )


def compute_direction(x, sigma, A, b, y, ATy, lambdas, cg_param, solver, fit_intercept):

    x_tilde = x / sigma - ATy

    x_tilde_prox = x - sigma * ATy
    x_tilde_prox[fit_intercept:] = prox_slope2(
        x_tilde_prox[fit_intercept:], sigma * lambdas
    )

    nabla_psi = y + b - A @ x_tilde_prox

    W = build_W(x_tilde, sigma, lambdas, A, fit_intercept)

    m, r1_plus_r2 = W.shape

    if solver == "auto":
        if r1_plus_r2 <= 100 * m:
            solver = "woodbury"
        elif m > 10_000 and m / r1_plus_r2 > 0.1:
            solver = "cg"
        else:
            solver = "standard"

    if solver == "woodbury":
        # Use Woodbury factorization solve
        WTW = W.T @ W
        if sparse.issparse(A):
            V_inv = sparse.eye(m, format="csc") - W @ spsolve(
                sparse.eye(r1_plus_r2, format="csc") + WTW, W.T
            )
        else:
            # print(WTW)
            np.fill_diagonal(WTW, WTW.diagonal() + 1)
            V_inv = -(W @ solve(WTW, W.T))
            np.fill_diagonal(V_inv, V_inv.diagonal() + 1)

        d = V_inv @ (-nabla_psi)
    elif solver == "cg":
        # Use conjugate gradient
        V = W @ W.T
        if sparse.issparse(A):
            V += sparse.eye(m, format="csc")
        else:
            np.fill_diagonal(V, V.diagonal() + 1)

        rel_tol = cg_param["abs_tol"]
        abs_tol = cg_param["rel_tol"]

        # preconditioner
        M = sparse.diags(V.diagonal())

        d, _ = cg(V, -nabla_psi, tol=rel_tol, atol=abs_tol, M=M)
    else:
        V = W @ W.T
        if sparse.issparse(A):
            V = sparse.eye(V.shape[0]) + W @ W.T
            d = spsolve(V, -nabla_psi)
        else:
            np.fill_diagonal(V, V.diagonal() + 1)
            d = cho_solve(cho_factor(V), -nabla_psi)

    return d, nabla_psi


@njit
def line_search(
    y, d, x, ATy, ATd, b, lambdas, sigma, nabla_psi, delta, mu, beta, fit_intercept
):
    # step 1b, line search
    mj = 0

    psi0 = psi(y, x, ATy, b, lambdas, sigma, fit_intercept)
    nabla_psi_d = nabla_psi @ d
    while True:
        alpha = delta**mj
        lhs = psi(y + alpha * d, x, ATy + alpha * ATd, b, lambdas, sigma, fit_intercept)
        rhs = psi0 + mu * alpha * nabla_psi_d

        if lhs <= rhs:
            break

        mj = 1 if mj == 0 else mj * beta

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
    fit_intercept,
):
    sigma = local_param["sigma"]

    d, nabla_psi = compute_direction(
        x_old, sigma, A, b, y, ATy, lambdas, cg_param, solver, fit_intercept
    )
    ATd = A.T @ d
    alpha = line_search(
        y,
        d,
        x_old,
        ATy,
        ATd,
        b,
        lambdas,
        sigma,
        nabla_psi,
        line_search_param["delta"],
        line_search_param["mu"],
        line_search_param["beta"],
        fit_intercept,
    )

    # step 1c, update y
    y += alpha * d
    ATy += alpha * ATd

    # step 2, update x
    x = x_old - sigma * ATy
    x[fit_intercept:] = prox_slope2(x[fit_intercept:], sigma * lambdas)

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
    fit_intercept=True,
    max_inner_it=100_000,
    solver="auto",
    line_search_param={"mu": 0.2, "delta": 0.5, "beta": 2},
    cg_param={"rel_tol": 1e-5, "abs_tol": 1e-8},
    local_param={"epsilon": 1.0, "delta": 1.0, "delta_prime": 1.0, "sigma": 0.5},
    gap_freq=1,
    tol=1e-6,
    max_epochs=1000,
    max_time=np.inf,
    verbose=True,
):
    if solver not in ["auto", "standard", "woodbury", "cg"]:
        raise ValueError("`solver` must be one of auto, standard, woodbury, and cg")

    m, n = A.shape

    lambdas = lambdas.copy() * m

    if fit_intercept:
        A = add_intercept_column(A)
        n += 1

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
                fit_intercept,
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
            theta /= max(1, dual_norm_slope(A[:, fit_intercept:], theta, lambdas / m))

            primal = (0.5 / m) * norm(r) ** 2 + np.sum(
                lambdas * np.sort(np.abs(x[fit_intercept:]))[::-1]
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

    intercept = x[0] if fit_intercept else 0.0

    return x[fit_intercept:], intercept, primals, gaps, times
