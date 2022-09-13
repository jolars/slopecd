# -*- coding: utf-8 -*-

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use the ADMM to solve the SLOPE model:
%% min { 0.5*||Ax - b||^2 + kappa_lambda (x) 
%% from the dual pespective: 
%% - min {0.5*||xi||^2 + b'*xi + kappa^*_lambda (u): A'*xi + u = 0}
%% where kappa_lambda(x)=lambda'*sort(abs(x),'descend')  
%% and kappa^*_lambda is the conjugate function, [m,n]=size(A) with m<n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


import time
from typing import Dict  # 判断变量类型是否为Dict

import numpy as np

from slope.solvers.newt_alm_utils.Solvers import (
    cardcal,
    mychol,
    mylinsysolve,
    proxSortedL1,
    psqmry,
)


def sigma_fun(iterk):
    if iterk < 30:
        sigma_update_iter = 3
    elif iterk < 60:
        sigma_update_iter = 6
    elif iterk < 120:
        sigma_update_iter = 12
    elif iterk < 250:
        sigma_update_iter = 25
    elif iterk < 500:
        sigma_update_iter = 50
    elif iterk < float("inf"):  # better than (iter < 1000)
        sigma_update_iter = 100
    return sigma_update_iter


def ADMM(Ainput, b, n, lambda1, options, x0, xi0, u0):
    eps = 2.2204e-16
    if "sigma" in options.__dict__:
        sigma = options.sigma
    else:
        sigma = 1

    if "gamma" in options.__dict__:
        gamma = options.gamma
    else:
        gamma = 1.618

    if "stoptol" in options.__dict__:
        stoptol = options.stoptol
    else:
        stoptol = 1e-6

    if "printyes" in options.__dict__:
        printyes = options.printyes
    else:
        printyes = 0

    if "maxiter" in options.__dict__:
        maxiter = options.maxiter
    else:
        maxiter = 20000

    if "printminoryes" in options.__dict__:
        printminoryes = options.printminoryes
    else:
        printminoryes = 0

    if "sig_fix" in options.__dict__:
        sig_fix = options.sig_fix
    else:
        sig_fix = 0

    if "dscale" in options.__dict__:
        dscale = options.dscale

    if "rescale" in options.__dict__:
        rescale = options.rescale
    else:
        rescale = 1

    if "use_infeasorg" in options.__dict__:
        use_infeasorg = options.use_infeasorg
    else:
        use_infeasorg = 0

    if "phase2" in options.__dict__:
        phase2 = options.phase2
    else:
        phase2 = 0

    if "fusedyes" in options.__dict__:
        fusedyes = options.fusedyes
    else:
        fusedyes = 1

    if "sGS" in options.__dict__:
        sGS = options.sGS

    if "Asolver" in options.__dict__:
        Asolver = options.Asolver
    else:
        Asolver = "prox"
    stopop = 0
    gapcon = 1
    pdconst = 1

    """ Amap and ATmap """
    tstart = time.time()
    tstart_cpu = time.perf_counter()
    m = len(b)
    if isinstance(Ainput, Dict):
        A0 = Ainput["A"]
        Amap0 = Ainput["Amap"]
        ATmap0 = Ainput["ATmap"]
    else:
        A0 = Ainput
        Amap0 = lambda x: A0 @ x
        ATmap0 = lambda y: A0.T @ y
    AATmap0 = lambda x: Amap0(ATmap0(x))
    Amap = Amap0
    ATmap = ATmap0
    AATmap = AATmap0

    class eigsopt:
        pass

    if Asolver == "prox":
        rA = 1
        AATmap0 = A0 @ A0.T
        Lipeig0, Lipvector0 = np.linalg.eig(AATmap0)
        Lipsort = np.argsort(-Lipeig0)
        dA = Lipeig0[list(Lipsort[0:rA])]
        VA = Lipvector0[:, list(Lipsort[0:rA])]
        dA = np.diag(dA)
        rA = sum(sum(dA > 0))
        if printyes:
            for i in range(rA):
                print("{}-th eigen = {:.2e}".format(i + 1, dA[i, i]))
        proxA = min(10, rA)
        dA = dA[0:proxA]
        VA = VA[:, 0:proxA]
        VAt = VA.T
        MAmap = lambda xi: dA[-1, -1] * xi + VA @ ((dA - dA[-1, -1]).T * (VAt @ xi))
        MAinv = lambda xi, sigma: xi / (1 + sigma * dA[-1, -1]) + VA[:, 0:proxA] @ (
            (1 / (1 + sigma * dA[0:proxA]) - 1 / (1 + sigma * dA[-1, -1]))
            * (VAt[0:proxA, :] @ xi)
        )
        pdconst = 5
    elif Asolver == "direct" and "A0" in dir():
        AAt0 = A0 @ A0.T
    """ initiallization """
    borg = b
    lambda1org = lambda1
    normborg = 1 + np.linalg.norm(borg)
    normb = normborg
    if "x0" not in dir() or "xi0" not in dir() or "u0" not in dir():
        x = np.zeros([n, 1])
        xi = np.zeros([m, 1])
        u = np.zeros([n, 1])
    else:
        x = x0
        xi = xi0
        u = u0
    bscale = 1
    cscale = 1
    objscale = bscale * cscale
    if printyes:
        print(' ****************************************************************************')
        print('\n \t\t  Phase I:  ADMM  for solving SLOPE with  gamma = {:.3f}'.format(gamma))
        print('\n ***************************************************************************')
        if printminoryes:
            print('\n problem size: n = %3.0f, nb = %3.0f',n, m)
            print('\n ---------------------------------------------------')
        print('\n  iter|  [pinfeas  dinfeas] [pinforg dinforg]   relgap |    pobj       dobj      |  time |  sigma  |gamma |')
    Atxi = ATmap(xi)
    AAtxi = Amap(Atxi)
    if Asolver == "cg":
        IpsigAAtxi = xi + sigma * AAtxi
    elif "AAt0" in dir():
        AAt = AAt0
        Lxi = mychol.mychol(np.eye(m) + sigma * AAt, m)
    Ax = Amap(x)
    Rp1 = Ax - b
    Rd = Atxi + u
    ARd = Amap(Rd)
    primfeas = np.linalg.norm(Rp1 - xi) / normborg
    dualfeas = np.linalg.norm(Rd) / (1 + np.linalg.norm(u))
    maxfeas = max(primfeas, dualfeas)
    primfeasorg = primfeas
    dualfeasorg = dualfeas
    maxfeasorg = maxfeas

    class runhist:
        pass

    runhist.dualfeas = []
    runhist.primfeas = []
    runhist.dualfeasorg = []
    runhist.primfeasorg = []
    runhist.maxfeasorg = []
    runhist.sigma = []
    runhist.cputime = []
    runhist.psqmrxiiter = []
    runhist.xr = []
    runhist.feasratioorg = []
    runhist.time = []
    runhist.cputime.append(time.time() - tstart)
    runhist.primobj = []
    runhist.dualobj = []
    runhist.relgap = []
    runhist.psqmrxiiter.append(0)
    '''注释掉print'''
    if printyes:
        print( 'initial primfeasorg = {:.2e}, dualfeasorg = {:.2e}'.format(primfeasorg, dualfeasorg) )
    """ main Loop """
    breakyes = 0
    prim_win = 0
    dual_win = 0
    repeaty = 0
    msg = []
    admm_sb_c = 1
    for iterk in range(maxiter):
        if (rescale >= 3 and ((iterk + 1) % 203)) == 0 or iterk == 0:
            normAtxi = np.linalg.norm(Atxi)
            normx = np.linalg.norm(x)
            normu = np.linalg.norm(u)
            normuxi = max([normAtxi, normu])
        if (
            (
                (rescale == 1)
                and (maxfeas < 5e2)
                and (iterk > 20)
                and (abs(relgap) < 0.5)
            )
            or (
                (rescale == 2)
                and (maxfeas < 1e-2)
                and (abs(relgap) < 0.05)
                and (iterk > 39)
            )
            or (
                (rescale >= 3)
                and (max(normx / normuxi, normuxi / normx) > 1.2)
                and ((iterk + 1) % 203 == 0)
            )
        ):
            if rescale <= 2:
                normAtxi = np.linalg.norm(Atxi)
                normx = np.linalg.norm(x)
                normu = np.linalg.norm(u)
                normuxi = max([normAtxi, normu])
            const = 1
            bscale2 = normx * const
            cscale2 = normuxi * const
            sbc = np.sqrt(bscale2 * cscale2)
            b = b / sbc
            u = u / cscale2
            lambda1 = lambda1 / cscale2
            x = x / bscale2
            xi = xi / sbc
            Rp1 = Rp1 / sbc
            admm_sb_c = admm_sb_c * np.sqrt(bscale2 / cscale2)
            Amap = lambda x: A0 @ x * admm_sb_c
            ATmap = lambda x: A0.T @ x * admm_sb_c
            AATmap = lambda x: (A0 @ A0.T) @ x * admm_sb_c
            Ax = Ax / sbc
            ARd = ARd * np.sqrt(bscale2 / cscale2**3)
            if "AAt" in dir():
                AAt = (bscale2 / cscale2) * AAt
            if Asolver == "cg":
                IpsigAAtxi = IpsigAAtxi / sbc
            elif Asolver == "prox":
                dA = dA * bscale2 / cscale2
                MAmap = lambda xi: dA[-1, -1] * xi + VA @ (
                    (dA - dA[-1, -1]) * (VAt @ xi)
                )
                MAinv = lambda xi, sigma: xi / (1 + sigma * dA[-1, -1]) + VA[
                    :, 0:proxA
                ] @ (
                    (1 / (1 + sigma * dA[0:proxA]) - 1 / (1 + sigma * dA[-1, -1]))
                    * (VAt[0:proxA, :] @ xi)
                )
            sigma = sigma * (cscale2 / bscale2)
            cscale = cscale * cscale2
            bscale = bscale * bscale2
            objscale = objscale * (cscale2 * bscale2)
            normb = 1 + np.linalg.norm(b)
            '''注释掉print'''
            if printyes:
                print( '[rescale={:.0f} : {:.0f}| [{:.2e}  {:.2e}  {:.2e} | {:.2e} {:.2e}| {:.2e}]'\
            .format(rescale,iterk,normx,normAtxi,normu,bscale,cscale,sigma) )
            rescale = rescale + 1
            prim_win = 0
            dual_win = 0
        xiold = xi
        uold = u
        xold = x
        Axold = Ax
        """ compute xi """

        class parxi:
            pass

        if Asolver == "cg":
            rhsxi = Rp1 - sigma * (ARd - (IpsigAAtxi - xi) / sigma)
            parxi.tol = max(0.9 * stoptol, min(1 / (iterk + 1) ** 1.1, 0.9 * maxfeas))
            parxi.sigma = sigma
            xi, IpsigAAtxi, resnrmxi, solve_okxi = psqmry.psqmry(
                "matvecxi.matvecxi", AATmap, rhsxi, parxi, xi, IpsigAAtxi
            )
        elif Asolver == "prox":
            rhsxi = Rp1 - sigma * (ARd - MAmap(xi))
            xi = MAinv(rhsxi, sigma)
        elif Asolver == "direct":
            rhsxi = Rp1 - sigma * Amap(u)
            if m <= 300:
                xi = np.linalg.inv((np.eye(m) + sigma * AAt)) @ rhsxi
            else:
                xi = mylinsysolve.mylinsysolve(Lxi, rhsxi)
        Atxi = ATmap(xi)
        """ first time compute u """
        uinput = x - sigma * Atxi
        up, _ = proxSortedL1.proxSortedL1(uinput, sigma * lambda1)
        u = (uinput - up) / sigma
        """ update mutilplier Xi, y """
        Rd = Atxi + u
        x = xold - gamma * sigma * Rd
        ARd = Amap(Rd)
        Ax = Axold - gamma * sigma * ARd
        Rp1 = Ax - b
        normRp = np.linalg.norm(Rp1 - xi)
        normRd = np.linalg.norm(Rd)
        normu = np.linalg.norm(u)
        etaC = 0
        etaCorg = 0
        primfeas = normRp / normb
        dualfeas = max([normRd / (1 + normu), 1 * etaC])
        maxfeas = max(primfeas, dualfeas)
        dualfeasorg = max([normRd * cscale / (1 + normu * cscale), 1 * etaCorg])
        primfeasorg = np.sqrt(bscale * cscale) * normRp / normborg
        maxfeasorg = max(primfeasorg, dualfeasorg)
        """ record history """
        runhist.dualfeas.append(dualfeas)
        runhist.primfeas.append(primfeas)
        runhist.dualfeasorg.append(dualfeasorg)
        runhist.primfeasorg.append(primfeasorg)
        runhist.maxfeasorg.append(maxfeasorg)
        runhist.sigma.append(sigma)
        if Asolver == "cg":
            runhist.psqmrxiiter.append(len(resnrmxi) - 1)
        runhist.xr.append(sum(abs(x) > 1e-10))
        """ check for termination """
        if stopop == 1:
            if max([primfeasorg, dualfeasorg]) < 5 * stoptol:
                grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
                etaorg = np.linalg.norm(
                    x * bscale
                    - proxSortedL1.proxSortedL1(x * bscale - grad, lambda1org)
                )
                # our case
                eta = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x * bscale))
                if eta < stoptol:
                    breakyes = 1
                    msg = "converged"
                primobj = objscale * (
                    0.5 * np.linalg.norm(xi) ** 2
                    + lambda1.T @ sorted(abs(x), reverse=True)
                )
            if max([primfeasorg, dualfeasorg]) < stoptol:
                dualobj = objscale * (
                    -0.5 * np.linalg.norm(xi) ** 2
                    - b.T @ xi
                    + lambda1.T @ sorted(abs(up), reverse=True)
                    - up.T @ u
                )
                relgap = abs(primobj - dualobj) / max(1, abs(primobj))
                if abs(relgap) < stoptol and eta < np.sqrt(stoptol):
                    breakyes = 2
                    msg = "Converged"
            elif "optval" in options.__dict__:
                if primobj < options.optval:
                    breakyes = 3
                    msg = "Opt_value converged"
        elif stopop == 2:
            if max([primfeasorg, dualfeasorg]) < stoptol:
                primobj = objscale * (
                    0.5 * np.linalg.norm(xi) ** 2
                    + lambda1.T @ sorted(abs(x), reverse=True)
                )
                dualobj = objscale * (
                    -0.5 * np.linalg.norm(xi) ** 2
                    - b.T @ xi
                    + lambda1.T @ sorted(abs(x), reverse=True)
                    - up.T @ u
                )
                relgap = abs(primobj - dualobj) / max(1, abs(primobj))
                if abs(relgap) < gapcon * stoptol:
                    grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
                    etaorg = np.linalg.norm(
                        x * bscale
                        - proxSortedL1.proxSortedL1(x * bscale - grad, lambda1org)
                    )
                    eta = etaorg / (
                        1 + np.linalg.norm(grad) + np.linalg.norm(x * bscale)
                    )
                    if eta < max(0.01, np.sqrt(stoptol)):
                        breakyes = 88
                        msg = "Converged"
        elif stopop == 0:
            if max([primfeasorg, dualfeasorg]) < pdconst * stoptol:
                if "eta" not in dir() or (iterk + 1) % 50 == 1 or eta < 1.2 * stoptol:
                    grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
                    xbscalegrad, _ = proxSortedL1.proxSortedL1(
                        x * bscale - grad, lambda1org
                    )
                    etaorg = np.linalg.norm(x * bscale - xbscalegrad)
                    eta = etaorg / (
                        1 + np.linalg.norm(grad) + np.linalg.norm(x * bscale)
                    )
                    gs = sorted(abs(grad), reverse=True)
                    gs_lambda1org = gs - lambda1org
                    infeas = max(max(gs_lambda1org.cumsum()), 0) / lambda1org[0]
                    relgap = abs(primobj - dualobj) / max(1, abs(primobj))
                    if eta < stoptol and infeas < stoptol and relgap < stoptol:
                        breakyes = 1
                        msg = "Strongly converged"
        if time.time() - tstart > 3 * 3600:
            breakyes = 777
            msg = "3 hours,time out!"
        """ print results """
        if iterk <= 200 - 1:
            print_iter = 20
        elif iterk <= 2000 - 1:
            print_iter = 100
        else:
            print_iter = 200
        if (((iterk + 1) % print_iter) == 1 or iterk == maxiter - 1) or (breakyes):
            primobj = objscale * (
                0.5 * np.linalg.norm(xi) ** 2 + lambda1.T @ sorted(abs(x), reverse=True)
            )
            dualobj = objscale * (-0.5 * np.linalg.norm(xi) ** 2 - b.T @ xi)
            relgap = abs(primobj - dualobj) / max(1, abs(primobj))
            ttime = time.time() - tstart
            if (printyes):
                print(' {}| [{:.2e} {:.2e}] [{:.2e}  {:.2e}]  {:.2e}| {:.3e}  {:.3e} |   {:.1f}|  {:.2e}|  {:.3f}| '.format(iterk,primfeas,dualfeas,primfeasorg, dualfeasorg,relgap[0][0],primobj[0][0],dualobj[0][0],ttime, sigma,gamma))
                if Asolver=='cg':
                    print('[{}  {}]'.format(len(resnrmxi)-1, solve_okxi))
            print_iter5 = 5 * print_iter
            if ((iterk + 1) % (print_iter5)) == 1:
                normx = np.linalg.norm(x)
                normAtxi = np.linalg.norm(Atxi)
                normu = np.linalg.norm(u)
                if (printyes):
                    print( '[normx,Atxi,u ={:.2e}  {:.2e} {:.2e}]'.format(normx,normAtxi, normu) )
            runhist.primobj.append(primobj)
            runhist.dualobj.append(dualobj)
            runhist.time.append(ttime)
            runhist.relgap.append(relgap)
        if breakyes > 0:
            if printyes:
                print("\n  breakyes = %3.1f, %s", breakyes, msg)
            break
        """ update sigma """
        if maxfeas < 5 * stoptol:
            use_infeasorg = 1
        if use_infeasorg:
            feasratio = primfeasorg / dualfeasorg
            runhist.feasratioorg.append(feasratio)
        else:
            feasratio = primfeas / dualfeas
            runhist.feasratioorg.append(feasratio)
        if feasratio < 1:
            prim_win = prim_win + 1
        else:
            dual_win = dual_win + 1
        sigma_update_iter = sigma_fun(iterk + 1)
        sigmascale = 1.25
        sigmaold = sigma
        if not (sig_fix) and (((iterk + 1) % sigma_update_iter) == 0):
            sigmamax = 1e6
            sigmamin = 1e-8
            if iterk <= 2500 - 1:
                if prim_win > max(1, 1.2 * dual_win):
                    prim_win = 0
                    sigma = min(sigmamax, sigma * sigmascale)
                elif dual_win > max(1, 1.2 * prim_win):
                    dual_win = 0
                    sigma = max(sigmamin, sigma / sigmascale)
            else:
                1 == 1
                if use_infeasorg:
                    feasratiosub = runhist.feasratioorg[max(1, iterk - 19 + 1) : iterk]
                else:
                    feasratiosub = runhist.feasratio[max(1, iterk - 19 + 1) : iterk]
                meanfeasratiosub = np.mean(feasratiosub)
                if meanfeasratiosub < 0.1 or meanfeasratiosub > 1 / 0.1:
                    sigmascale = 1.4
                elif meanfeasratiosub < 0.2 or meanfeasratiosub > 1 / 0.2:
                    sigmascale = 1.35
                elif meanfeasratiosub < 0.3 or meanfeasratiosub > 1 / 0.3:
                    sigmascale = 1.32
                elif meanfeasratiosub < 0.4 or meanfeasratiosub > 1 / 0.4:
                    sigmascale = 1.28
                elif meanfeasratiosub < 0.5 or meanfeasratiosub > 1 / 0.5:
                    sigmascale = 1.26
                primidx = [i for (i, val) in enumerate(feasratiosub[0]) if val <= 1]
                dualidx = [i for (i, val) in enumerate(feasratiosub[0]) if val > 1]
                if len(primidx) >= 12:
                    sigma = min(sigmamax, sigma * sigmascale)
                if len(dualidx) >= 12:
                    sigma = max(sigmamin, sigma / sigmascale)
        if abs(sigmaold - sigma) > eps:
            if Asolver == "cg":
                parxi.sigma = sigma
                AAtxi = (IpsigAAtxi - xi) / sigmaold
                IpsigAAtxi = xi + sigma * AAtxi
            elif "Lxi" in dir():
                Lxi = mychol.mychol(np.eye(m) + sigma * AAt, m)
    """ recover orignal variables """

    class info:
        pass

    if iterk == maxiter - 1:
        msg = " maximum iteration reached"
        info.termcode = 3
    xi = xi * np.sqrt(bscale * cscale)
    Atxi = ATmap0(xi)
    x = x * bscale
    u = u * cscale
    if "up" not in dir():
        up = x
    up = up * bscale
    Ax = Ax * np.sqrt(bscale * cscale)
    Rd = Atxi + u
    Rp1 = Ax - borg
    normRp = np.linalg.norm(Rp1 - xi)
    normRd = np.linalg.norm(Rd)
    normu = np.linalg.norm(u)
    primfeasorg = normRp / normborg
    dualfeasorg = normRd / (1 + normu)
    primobj = 0.5 * np.linalg.norm(xi) ** 2 + lambda1org.T @ sorted(
        abs(x), reverse=True
    )
    dualobj = -(0.5 * np.linalg.norm(xi) ** 2 + borg.T @ xi)
    primobjorg = 0.5 * np.linalg.norm(Ax - borg) ** 2 + lambda1org.T @ sorted(
        abs(x), reverse=True
    )
    relgap = abs(primobj - dualobj) / max(1, abs(primobj))
    obj = [primobj, dualobj]
    if iterk + 1 > 0:
        grad = ATmap0(Ax - borg)
        xbscale_grad, _ = proxSortedL1.proxSortedL1(x * bscale - grad, lambda1org)
        etaorg = np.linalg.norm(x * bscale - xbscale_grad)
        eta = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x))
    else:
        etaorg = float("nan")
        eta = float("nan")
    runhist.m = m
    runhist.n = n
    ttime = time.time() - tstart
    ttime_cpu = time.perf_counter() - tstart_cpu
    ttCG = sum(runhist.psqmrxiiter)
    runhist.iter = iterk
    runhist.totaltime = ttime
    runhist.primobjorg = primobj
    runhist.dualobjorg = dualobj
    runhist.maxfeas = max([dualfeasorg, primfeasorg])
    runhist.etaorg = etaorg
    runhist.eta = eta
    info.m = m
    info.n = n
    info.minx = min(min(x))
    info.maxx = max(max(x))
    info.relgap = relgap
    info.ttCG = ttCG
    info.iter = iterk
    info.time = ttime
    info.time_cpu = ttime_cpu
    info.sigma = sigma
    info.etaorg = etaorg
    info.eta = eta
    info.bscale = bscale
    info.cscale = cscale
    info.objscale = objscale
    info.dualfeasorg = dualfeasorg
    info.primfeasorg = primfeasorg
    info.obj = obj
    info.nnzx = cardcal.cardcal(x, 0.999, 1e-16)
    if phase2 == 1:
        info.Ax = Ax
        info.Atxi = Atxi
    if printminoryes:
        if not msg:
            print("%s", msg)
        print("--------------------------------------------------------------")
        print("number iter = %2.0d", iterk)
        print("time = %3.2f", ttime)
        if iterk >= 1:
            print("\n  time per iter = %5.4f", ttime / iterk)
        print("\n  cputime = %3.2f", ttime_cpu)
        print(
            "\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e",
            primobj,
            dualobj,
            relgap,
        )
        print("\n  primobjorg = %10.9e", primobjorg)
        print("\n  primfeasorg  = %3.2e, dualfeasorg = %3.2e", primfeasorg, dualfeasorg)
        if iterk >= 1:
            print(
                "\n  Total CG number = %3.0d, CG per iter = %3.1f", ttCG, ttCG / iterk
            )
        print(" eta = %3.2e, etaorg = %3.2e", eta, etaorg)
        print("\n  min(X)    = %3.2e, max(X)    = %3.2e", info.minx, info.maxx)
        print(
            "\n  number of nonzeros in x (0.999) = %3.0d",
            cardcal.cardcal(x, 0.999, 1e-6),
        )
        print("\n--------------------------------------------------------------")

    return obj, xi, u, x, info, runhist


# ---------------------------------------------
