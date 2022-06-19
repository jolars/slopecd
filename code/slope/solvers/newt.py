import warnings
from timeit import default_timer as timer

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from numpy.random import default_rng
from scipy import sparse, stats
from scipy.linalg import inv, norm, solve
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from slope.solvers import prox_grad
from slope.utils import dual_norm_slope, prox_slope
from slope.clusters import get_clusters


def nonzero_sign(x):
    n = len(x)
    out = np.empty(n)

    for i in range(n):
        s = np.sign(x[i])
        out[i] = s if s != 0 else 0

    return out


def permutation_matrix(x):
    n = len(x)

    signs = nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = sparse.lil_matrix((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[ord_j]

    return sparse.csc_array(pi)


def psi(y, x, A, b, lambdas, sigma):
    ATy = A.T @ y

    # TODO(jolars): not at all sure what epsilon and lambdas should be here
    # epsilon = np.sum(np.abs(ATy))
    w = x - sigma * ATy
    # u = sorted_l1_proj(x_tilde, lambdas / sigma, epsilon / sigma)
    u = (1 / sigma) * (w - prox_slope(w, lambdas * sigma))

    phi = 0.5 * norm(u - w / sigma) ** 2

    return 0.5 * norm(y) ** 2 + b @ y - (0.5 / sigma) * norm(x) ** 2 + sigma * phi


def compute_direction(x, sigma, A, y, lambdas, cg_param):
    
    
    debug = False
    
    B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
    # step 1a, compute the newton direction
    x_tilde = x / sigma - (A.T @ y)

    # construct M
    pi = permutation_matrix(x_tilde)  # pi @ x == np.sort(np.abs(x))[::-1]

    # construct Jacobian
    # P = jacobian(pi @ w, pi, B, lambdas)
    ord = np.argsort(np.abs(x_tilde))[::-1]

    x_lambda = np.abs(prox_slope(x_tilde, sigma)[ord])
    z = spsolve(B @ B.T, B @ (np.abs(x_tilde[ord]) - lambdas - x_lambda))

    if debug:
        print(f"x_tilde: {x_tilde}")
        print(f"x_lambda: {x_lambda}")
        print(f"z: {z}")

    z_supp = np.where(z != 0)[0]
    I_x_lambda = np.where(B @ x_lambda == 0)[0]

    Gamma = np.intersect1d(z_supp, I_x_lambda)

    B_Gamma = B[Gamma, :]

    P = sparse.eye(n, format="csc") - B_Gamma.T @ spsolve(
        B_Gamma @ B_Gamma.T, B_Gamma
    )

    # # alternate way of constructing P
    # c, c_ptr, c_ind, N = get_clusters(x_tilde)
    # H_list = []
    # for i in range(N):
    #     N_i = c_ptr[i + 1] - c_ptr[i]
    #     H_i = np.ones((N_i, N_i), dtype=int) if c[i] != 0 else sparse.csc_matrix((N_i, N_i), dtype=int)
    #     H_list.append(H_i)

    # H = sparse.block_diag(H_list, format="csc")
        
    # # a = 
    # V1 = A @ (pi[a, :].T)

    M = pi.T @ P @ pi

    V = sigma * (A @ M @ A.T)
    np.fill_diagonal(V, V.diagonal() + 1.0)

    nabla_psi = y + b - A @ prox_slope(x - sigma * (A.T @ y), sigma * lambdas)

    d = solve(V, -nabla_psi)

    if norm(V @ d + nabla_psi) > min(cg_param['eta'], norm(nabla_psi) ** (1 + cg_param['tau'])):
        warnings.warn("Solver did not work")

    if debug:
        print("P\n", P)
        print("M\n", M)
        print("d\n", d)
        print("nabla_psi\n", nabla_psi)
        print("norm(nabla_psi)\n", norm(nabla_psi))

    return d, nabla_psi


def line_search(y, d, x, A, b, lambdas, sigma, nabla_psi, line_search_param): 
    # step 1b, line search
    mj = 0
    
    psi0 = psi(y, x, A, b, lambdas, sigma)
    while True:
        alpha = line_search_param['delta']**mj
    
        lhs = psi(y + alpha * d, x, A, b, lambdas, sigma)
        rhs = psi0 + line_search_param['mu'] * alpha * nabla_psi @ d
    
    
        if lhs <= rhs:
            break
    
        mj = 1 if mj == 0 else mj * line_search_param['beta']
    
    return alpha

def check_convegence(x_diff_norm, 
                     nabla_psi, 
                     epsilon_k, 
                     sigma,
                     delta_k,
                     delta_prime_k):
    
    # check for convergence
    norm_nabla_psi = norm(nabla_psi)


    crit_A = norm_nabla_psi <= epsilon_k / np.sqrt(sigma)
    crit_B1 = norm_nabla_psi <= (delta_k / np.sqrt(sigma)) * x_diff_norm
    crit_B2 = norm_nabla_psi <= (delta_prime_k / sigma) * x_diff_norm

    if crit_A and crit_B1 and crit_B2:
        return True
    
    return False

def inner_step(A,
               b,
               x,
               y,
               lambdas,
               x_old,
               local_param,
               line_search_param,
               cg_param):

    sigma = local_param['sigma']
    d, nabla_psi = compute_direction(x, sigma, A, y, lambdas,cg_param)
    alpha = line_search(y, d, x, A, b, lambdas, sigma, nabla_psi, line_search_param)
    
    # step 1c, update y
    y = y + alpha * d
    
    
    # step 2, update x
    x = prox_slope(x - sigma * (A.T @ y), sigma * lambdas)
    
    # check for convergence
    x_diff_norm = norm(x - x_old)
    
    
    
    converged = check_convegence(x_diff_norm, 
                                 nabla_psi, 
                                 local_param['epsilon'], 
                                 sigma,
                                 local_param['delta'],
                                 local_param['delta_prime'])
    return converged, x, y


def newton_solver(A,
                  b,
                  lambdas,
                  x = None,
                  y = None,
                  optim_param = {'max_epochs':100,
                                 'max_inner_it':10000,
                                 'tol':1e-8,
                                 'gap_freq':1},
                  line_search_param = {'mu':0.2,
                                       'delta':0.5,
                                       'beta':2},
                  cg_param = {'eta':1e-4,
                              'tau':0.5},
                  verbose =True):
    
    
    m, n = A.shape
    
    
    if x is None:
        x = rng.standard_normal(n)
    if y is None:
        y = rng.standard_normal(m)
    max_epochs   = optim_param['max_epochs']
    max_inner_it = optim_param['max_inner_it']

  
    
    # step 1 update parameters
    local_param = {'epsilon':0.1,
                   'delta':0.1,
                   'delta_prime':0.1,
                   'sigma':1}
    
    
    r = b.copy()
    theta = np.zeros(m)
    primals, gaps = [], []
    primals.append(norm(b) ** 2 / (2 * m))
    gaps.append(primals[0])
    
    
    
    
    for epoch in range(max_epochs):
        # step 1
        local_param['delta_prime'] *= 0.9
        local_param['epsilon']     *= 0.9
        local_param['delta']       *= 0.9
    
        x_old = x.copy()
    
        for j in range(max_inner_it):
            
            converged, x, y =  inner_step(A,
                                          b,
                                          x,
                                          y,
                                          lambdas,
                                          x_old,
                                          local_param,
                                          line_search_param,
                                          cg_param)
            
            
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
        local_param['sigma'] *= 1.1
    
    
        if epoch % optim_param['gap_freq'] == 0 :
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
    
            if gap < optim_param['tol']:
                break

    return x, gaps, primals


if __name__ =="__main__":         
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
    
    
    x = newton_solver(A,
                      b,
                      lambdas)
    

