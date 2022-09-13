"""
%%*************************************************************************
%% psqmr:  preconditioned symmetric QMR with left (symmetric) preconditioner. 
%%
%% b = rhs vector.
%% resnrm = norm of qmr-generated residual vector b-Ax. 
%%
%% SDPNAL: 
%% Copyright (c) 2008 by
%% Xinyuan Zhao, Defeng Sun, and Kim-Chuan Toh 
%%*************************************************************************
"""
import numpy as np
from numba import njit

import slope.solvers.newt_alm_utils.Solvers.mylinsysolve


def precondfun(par, r):
    precond = 0
    if par.precond:
        precond = par.precond
    if precond == 0:
        q = r
    elif precond == 1:
        q = par.invdiagM * r
    elif precond == 2:
        dlast = par.d[-1]
        tmp1 = 2 / dlast
        tmp2 = 1 / par.d - tmp1
        q = tmp1 * r + par.V * (tmp2 * (par.Vt * r))
    elif precond == 3:
        q = par.invM[r]
    elif precond == 4:
        q = mylinsysolve.mylinsysolve(par.L, r)
    return q


def psqmry(matvecfname, A, b, par, x0, Ax0):
    resnrm = []
    N = len(b)
    maxit = max(5000, np.sqrt(N))
    tol = 1e-6 * np.linalg.norm(b)
    stagnate_check = 20
    miniter = 0
    x0 = np.zeros([N, 1])
    #  if (nargin < 5); x0 = zeros(N,1); end
    if par.maxit:
        maxit = par.maxit
    if par.tol:
        tol = par.tol
    if par.stagnate_check_psqmr:
        stagnate_check = par.stagnate_check_psqmr
    if par.minitpsqmr:
        miniter = par.minitpsqmr
    solve_ok = 1
    printlevel = 0
    x = x0
    if np.linalg.norm(x) > 0:
        Aq = np.zeros([N, 1])
    r = b - Aq
    err = np.linalg.norm(r)
    resnrm[0] = err
    minres = err
    q = precondfun(par, r)
    tau_old = np.linalg.norm(q)
    rho_old = r.T @ q
    theta_old = 0
    d = np.zeros([N, 1])
    res = r
    Ad = np.zeros([N, 1])
    """  main loop """
    tiny = 1e-30
    for iterk in range(maxit):
        Aq = feval(matvecfname, q, par, A)
        sigma = q.T @ Aq
        if abs(sigma) < tiny:
            solve_ok = 2
            if printlevel:
                print("s1")
            break
        else:
            alpha = rho_old / sigma
            r = r - alpha * Aq
        u = precondfun(par, r)
        theta = np.linalg.norm(u) / tau_old
        c = 1 / np.sqrt(1 + theta**2)
        tau = tau_old * theta * c
        gam = c**2 * theta_old**2
        eta = c**2 * alpha
        d = gam * d + eta * q
        x = x + d
        """ stopping conditions  """
        Ad = gam * Ad + eta * Aq
        res = res - Ad
        err = np.linalg.norm(res)
        resnrm[iterk + 1] = err
        if err < minres:
            minres = err
        if (err < tol) and (iterk > miniter) and (b.Tx > 0):
            break
        if (iterk > stagnate_check) and (iterk > 10):
            ratio = resnrm[iterk - 9 : iterk + 1] / resnrm[iterk - 10 : iterk]
            if (min(ratio) > 0.997) and (max(ratio) < 1.003):
                if printlevel:
                    print("s")
                solve_ok = -1
                break
        if abs(rho_old) < tiny:
            solve_ok = 2
            print("s2")
            break
        else:
            rho = r.T @ u
            beta = rho / rho_old
            q = u + beta * q
        rho_old = rho
        tau_old = tau
        theta_old = theta
        if iterk == maxit:
            solve_ok = -2
        if solve_ok != -1:
            if printlevel:
                print(" ")
        Ax = b - res
    return x, Ax, resnrm, solve_ok
