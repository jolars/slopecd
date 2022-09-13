"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use Newt_ALM to solve the SLOPE model:
%% min { 0.5*||Ax - b||^2 + kappa_lambda (x) 
%% from the dual pespective: 
%% - min {0.5*||xi||^2 + b'*xi + kappa^*_lambda (u): A'*xi + u = 0}
%% where kappa_lambda(x)=lambda'*sort(abs(x),'descend')  
%% and kappa^*_lambda is the conjugate function, [m,n]=size(A) with m<n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

"""
% Algorithm 1: Newt_ALM
% Initialization: (1) by inputs x0,xi0,u0; 
%                 (2) by default x0,xi0,u0 are all zeros;
%                 (3) the numerical solution obtianed by ADMM with low
%                     accuracy in Phase I
% Augmented parameter sigma tuning: ranging from [sigmamin, sigmamax] and
%                                   scaled by factors sigmascale (=2) or
%                                   1/sigmascale_reduce (1/sqrt(2))
%                                   based on the changes of primal and
%                                   dual infeasibilities
% Stopping criteria: KKT residual; infeasibility; relative duality gap
%                    The accuracy parameter stoptol (1e-6 by default)
"""
import time
from typing import Dict

import numpy as np

from slope.solvers.newt_alm_utils.Solvers import cardcal, proxSortedL1
from slope.solvers.newt_alm_utils.WeightedK.admm import ADMM
from slope.solvers.newt_alm_utils.WeightedK.nal import NCGWK
from slope.utils import ConvergenceMonitor


def Newt_ALM(
    Ainput,
    b,
    n,
    lambda_lam,
    fit_intercept,
    options,
    x0="x0",
    xi0="xi0",
    u0="u0",
    gap_freq=1,
    tol=1e-6,
    max_epochs=1000,
    max_time=np.inf,
    verbose=False,
):
    if "maxiter" in options.__dict__:
        maxiter = options.maxiter
    else:
        maxiter = 5000

    if "stoptol" in options.__dict__:
        stoptol = options.stoptol
    else:
        stoptol = 1e-6

    if "printyes" in options.__dict__:
        printyes = options.printyes
    else:
        printyes = 0

    if "printminoryes" in options.__dict__:
        printminoryes = options.printminoryes
    else:
        printminoryes = 0

    if "rescale" in options.__dict__:
        rescale = options.rescale
    else:
        rescale = 1

    if "Lip" in options.__dict__:
        Lip = options.Lip

    if "precond" in options.__dict__:
        precond = options.precond
    else:
        precond = 2

    if "printsub" in options.__dict__:
        printsub = options.printsub
    else:
        printsub = 0

    if "runphaseI" in options.__dict__:
        runphaseI = options.runphaseI
    else:
        runphaseI = 0

    stopop = 2
    # gapcon          = 1

    Sd = lambda x: sorted(abs(x), reverse=True)
    startfun = ADMM.ADMM
    scale = 1

    class info:
        pass

    class admm_admm:
        pass

    admm_admm.iterk = 0
    admm_admm.time = 0
    admm_admm.timecpu = 0
    """  Amap and ATmap """
    tstart = time.time()
    tstart_cpu = time.perf_counter()
    m = len(b)

    if isinstance(Ainput, Dict):
        A = Ainput["A"]
        Amap0 = Ainput["Amap"]
        ATmap0 = Ainput["ATmap"]
    else:
        A = Ainput
        A_equal = A
        Amap0 = lambda x: A_equal @ x
        ATmap0 = lambda y: A_equal.T @ y
    # AATmap0           = lambda x: Amap0(ATmap0(x))

    intercept = 0.0

    A_submit = A
    sigmaLip = 1 / np.sqrt(Lip)
    lambda1org = lambda_lam
    borg = b
    normborg = 1 + np.linalg.norm(borg)
    # if x0 == 'x0' or xi0 == 'xi0' or u0 == 'u0':
    #     x             = np.zeros([n,1])
    #     xi            = np.zeros([m,1])
    #     u             = np.zeros([n,1])
    # else:
    x = x0
    xi = xi0
    u = u0

    A_tmp = A.copy()

    monitor = ConvergenceMonitor(
        A,
        borg,
        lambda1org / m,
        tol,
        gap_freq,
        max_time,
        verbose,
        intercept_column=fit_intercept,
    )

    """ phase I """

    class admm_op:
        pass

    admm_op.stoptol = 1e-2
    admm_op.maxiter = 100
    admm_op.sigma = min(1, sigmaLip)
    admm_op.phase2 = 1
    admm_op.use_infeasorg = 0
    obj = []
    if admm_op.maxiter > 0 and runphaseI:
        obj, xi, u, x, info_admm, runhist_admm = startfun(
            Ainput, b, n, lambda1org, admm_op, x, xi, u
        )

        admm_admm.xi0 = xi
        admm_admm.u0 = u
        admm_admm.x0 = x
        admm_admm.Atxi0 = info_admm.Atxi
        admm_admm.Ax0 = info_admm.Ax
        Atxi = admm_admm.Atxi0
        Ax = admm_admm.Ax0
        admm_admm.iterk = admm_admm.iterk + info_admm.iter
        admm_admm.time = admm_admm.time + info_admm.time
        admm_admm.timecpu = admm_admm.timecpu + info_admm.time_cpu
        bscale = info_admm.bscale
        cscale = info_admm.cscale
        objscale = info_admm.objscale
        if info_admm.eta < stoptol:
            # print("Problem solved in Phase I \n")
            # print(
            #     "ADMM Iteration No. = {:.0d}, ADMM time = {:.1f} s \n".format(
            #         admm_admm.iterk, admm_admm.time
            #     )
            # )
            info = info_admm
            info.m = m
            info.n = n
            info.minx = min(min(x))
            info.maxx = max(max(x))
            info.relgap = info_admm.relgap
            info.iterk = 0
            info.time = admm_admm.time
            info.time_cpu = admm_admm.timecpu
            info.admmtime = admm_admm.time
            info.admmtime_cpu = admm_admm.timecpu
            info.admmiter = admm_admm.iterk
            info.eta = info_admm.eta
            info.etaorg = info_admm.etaorg
            info.obj = obj
            info.maxfeas = max([info_admm.dualfeasorg, info_admm.primfeasorg])
            runhist = runhist_admm
            return
    else:
        Atxi = ATmap0(xi)
        Ax = Amap0(x)
        obj.append(0.5 * np.linalg.norm(Ax - borg) ** 2 + lambda1org.T @ Sd(x))
        obj.append(-0.5 * np.linalg.norm(xi) ** 2 + borg.T @ xi)
        bscale = 1
        cscale = 1
        objscale = bscale * cscale
    """   """
    if scale == 1:
        bsacle_const = bscale
        csacle_const = cscale
        b = b / np.sqrt(bscale * cscale)
        xi = xi / np.sqrt(bscale * cscale)
        bs_con_int = np.sqrt(bsacle_const / csacle_const)  # 传递
        Amap00 = lambda x: Amap0(x * np.sqrt(bsacle_const / csacle_const))
        ATmap00 = lambda x: ATmap0(x * np.sqrt(bsacle_const / csacle_const))
        if "A" in dir():
            A = A * np.sqrt(bscale / cscale)
        lambda1 = lambda_lam / cscale
        x = x / bscale
        u = u / cscale
        Ax = Ax / np.sqrt(bscale * cscale)
        Atxi = Atxi / cscale
        normb = 1 + np.linalg.norm(b)

    class Ainput_nal:
        pass

    Ainput_nal.Amap = Amap00
    Ainput_nal.ATmap = ATmap00
    if "A" in dir():
        Ainput_nal.A = A
    sigma = min(1, sigmaLip)
    Rp1 = Ax - b
    Rd = Atxi + u
    normu = np.linalg.norm(u)
    normRp = np.linalg.norm(Rp1 - xi)
    normRd = np.linalg.norm(Rd)
    primfeas = normRp / normb
    dualfeas = normRd / (1 + normu)
    maxfeas = max(primfeas, dualfeas)
    dualfeasorg = normRd * cscale / (1 + normu * cscale)
    primfeasorg = np.sqrt(bscale * cscale) * normRp / normborg
    maxfeasorg = max(primfeasorg, dualfeasorg)
    relgap = (obj[0] - obj[1]) / max(1, abs(obj[0]))

    class runhist:
        pass

    runhist.dualfeasorg = []
    runhist.dualfeas = []
    runhist.primfeas = []
    runhist.maxfeas = []
    runhist.primfeasorg = []
    runhist.dualfeasorg = []
    runhist.maxfeasorg = []
    runhist.sigma = []
    runhist.rank1 = []
    runhist.rank2 = []
    runhist.innerNT = []
    runhist.innerflsa = []
    runhist.xr = []
    runhist.primobj = []
    runhist.dualobj = []
    runhist.time = []
    runhist.relgap = []
    if printyes:
        print("**************************************************************")
        print("\t\t   Phase II: Newt_ALM ")
        print("**************************************************************")
        if printminoryes:
            print(" bscale = {:.2e},cscale = {:.2e}".format(bscale, cscale))
            print("**************************************************************")
        print(
            "  iter|  [pinfeas  dinfeas]  [pinforg  dinforg]    relgaporg|    pobj          dobj    |",
            end="",
        )
        print(" time | sigma |")
        print("**************************************************************")
        print(
            "# {}|  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}  {:.5e} {:.5e} {:.1f}  {:.2e} ".format(
                0,
                primfeas,
                dualfeas,
                primfeasorg,
                dualfeasorg,
                relgap[0][0],
                obj[0][0][0],
                obj[1][0][0],
                time.time() - tstart,
                sigma,
            )
        )
    """ %% semi-smooth Newton conjugate gradient (ssncg) method """
    SSNCG = 1

    class parNCG:
        pass

    if SSNCG:
        parNCG.matvecfname = "mvSLOPE"
        parNCG.sigma = sigma
        parNCG.tolconst = 0.5
        parNCG.n = n
        parNCG.precond = precond
    gamma = 1
    maxitersub = 10
    breakyes = 0
    prim_win = 0
    dual_win = 0
    RpGradratio = 1

    class ssncgop:
        pass

    ssncgop.tol = stoptol
    ssncgop.precond = precond
    ssncgop.bscale = bscale
    ssncgop.cscale = cscale
    ssncgop.printsub = printsub
    eta = 0
    etaorg = 0
    map_flag = 0
    sbc_submit = 1

    for iterk in range(maxiter):
        if (
            (rescale == 1) and (maxfeas < 5e2) and ((iterk % 3) == 0) and (iterk > 0)
        ) or (
            (rescale >= 2)
            and maxfeas < 1e-1
            and (abs(relgap) < 0.05)
            and (iterk >= 4)
            and (max(normx / normuxi, normuxi, normx) > 1.7)
            and (iterk % 5) == 0
        ):

            # % 表示取余数

            normAtxi = np.linalg.norm(Atxi)
            normx = np.linalg.norm(x)
            normu = np.linalg.norm(u)
            normuxi = max(normAtxi, normu)
            if normx < 1e-7:
                normx = 1
                normuxi = 1
            const = 1
            bscale2 = normx * const
            cscale2 = normuxi * const
            sbc = np.sqrt(bscale2 * cscale2)
            sb_c = np.sqrt(bscale2 / cscale2)
            sbc_submit = sbc_submit * sb_c
            b = b / sbc
            lambda1 = lambda1 / cscale2
            x = x / bscale2
            # Ainput_nal.Amap = lambda x:Ainput_nal.Amap(x*np.sqrt(bscale2/cscale2))
            class Ainput_nal_mid:
                pass

            map_flag = map_flag + 1
            if "A" in Ainput_nal.__dict__:
                Ainput_nal.A = Ainput_nal.A * np.sqrt(bscale2 / cscale2)
            if precond == 2 and "dA" in parNCG.__dict__:
                parNCG.dA = parNCG.dA * bscale2 / cscale2
            xi = xi / sbc
            Atxi = Atxi / cscale2
            Ax = Ax / sbc
            u = u / cscale2
            sigma = sigma * (cscale2 / bscale2)
            cscale = cscale * cscale2
            bscale = bscale * bscale2
            objscale = objscale * (cscale2 * bscale2)

            class ssncgop:
                pass

            ssncgop.bscale = bscale
            ssncgop.cscale = cscale
            normb = 1 + np.linalg.norm(b)
            """注释掉print"""
            if printyes:
                print(
                    "[rescale={:.0f}: {:.0f}| {:.2e} {:.2e} {:.2e} | {:.2e} {:.2e}| {:.2e}]".format(
                        rescale, iterk, normx, normAtxi, normu, bscale, cscale, sigma
                    )
                )
            rescale = rescale + 1
            prim_win = 0
            dual_win = 0

        xold = x
        uold = u
        parNCG.sigma = sigma
        parNCG.innerNT = 0
        parNCG.innerflsa = 0
        if dualfeas < 1e-5:
            maxitersub = max(maxitersub, 30)
        elif dualfeas < 1e-3:
            maxitersub = max(maxitersub, 30)
        elif dualfeas < 1e-1:
            maxitersub = max(maxitersub, 20)
        ssncgop.maxitersub = maxitersub
        # 新增
        Ainput_nal.A_submit_A = A_submit
        Ainput_nal.sbc_submit = sbc_submit
        Ainput_nal.bs_con_int = bs_con_int
        u, Atxi, xi, parNCG, runhist_NCG, info_NCG = NCGWK.NCGWK(
            b, Ainput_nal, x, Ax, Atxi, xi, lambda1, parNCG, ssncgop
        )
        if info_NCG.breakyes < 0:
            parNCG.tolconst = max(parNCG.tolconst / 1.06, 1e-3)
        x = info_NCG.up
        Ax = info_NCG.Aup
        Rd = Atxi + u
        Rp1 = Ax - b
        normRp = np.linalg.norm(Rp1 - xi)
        normRd = np.linalg.norm(Rd)
        normu = np.linalg.norm(u)
        primfeas = normRp / normb
        dualfeas = normRd / (1 + normu)
        maxfeas = max(primfeas, dualfeas)
        dualfeasorg = normRd * cscale / (1 + normu * cscale)
        primfeasorg = np.sqrt(bscale * cscale) * normRp / normborg
        maxfeasorg = max(primfeasorg, dualfeasorg)
        runhist.dualfeas.append(dualfeas)
        runhist.primfeas.append(primfeas)
        runhist.maxfeas.append(maxfeas)
        runhist.primfeasorg.append(primfeasorg)
        runhist.dualfeasorg.append(dualfeasorg)
        runhist.maxfeasorg.append(maxfeasorg)
        runhist.sigma.append(sigma)
        # runhist.rank1.append( sum(parNCG.info_u.rr1) )
        runhist.rank2.append(sum(parNCG.info_u.rr2))
        runhist.innerNT.append(parNCG.innerNT)
        runhist.innerflsa.append(parNCG.innerflsa)
        runhist.xr.append(sum(abs(x) > 1e-10))

        primobj = objscale * (0.5 * np.linalg.norm(xi) ** 2 + lambda1.T @ Sd(x))
        dualobj = objscale * (-0.5 * np.linalg.norm(xi) ** 2 - b.T @ xi)
        relgap = abs(primobj - dualobj) / max(1, abs(primobj))
        ttime = time.time() - tstart
        """ check for termination """
        if stopop == 1:
            if max([primfeasorg, dualfeasorg]) < 500 * max(1e-6, stoptol):
                grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
                etaorg = np.linalg.norm(
                    x * bscale
                    - proxSortedL1.proxSortedL1(x * bscale - grad, lambda1org)
                )
                """ proxSortedL1"""
                eta = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x * bscale))
                if eta < stoptol:
                    breakyes = 1
                    msg = "KKT residual converged"
                # elif abs(relgap) < stoptol and max([primfeasorg,dualfeasorg]) < stoptol and eta < np.sqrt(stoptol):
                #     breakyes = 2
                #     msg   = 'Relative gap & KKT residual converged'
        elif stopop == 2:
            if max([primfeasorg, dualfeasorg]) < 2 * stoptol:
                grad = ATmap0(Rp1 * np.sqrt(bscale * cscale))
                etaorg = np.linalg.norm(
                    x * bscale
                    - proxSortedL1.proxSortedL1(x * bscale - grad, lambda1org)[0]
                )
                eta = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x * bscale))
                gs = sorted(abs(grad), reverse=True)
                gs_lambda = gs - lambda1org
                infeas = max(max(gs_lambda.cumsum()), 0) / lambda1org[0]
                # if (eta< stoptol and abs(relgap) <stoptol  and infeas< stoptol):# stoptol
                #     breakyes = 999
                #     msg   = 'Relative gap & KKT residual converged'

        breakyes = monitor.check_convergence(bscale * x[fit_intercept:], intercept, iterk)

        if breakyes:
            break
        if printyes:
            print(
                "\n {}| [{:.2e}  {:.2e}] [{:.2e}  {:.2e}]  {:.2e}| {:.9e} {:.9e} | {:.1f} | {:.2e}|   {} | [{}]  sigmaorg = {:.2e}".format(
                    iterk + 1,
                    primfeas,
                    dualfeas,
                    primfeasorg,
                    dualfeasorg,
                    relgap[0][0],
                    primobj[0][0],
                    dualobj[0][0],
                    ttime,
                    sigma,
                    int(sum(parNCG.info_u.rr2)),
                    sum(abs(x) > 1e-10)[0],
                    sigma * (bscale / cscale),
                )
            )
            if "eta" in dir():
                print("\t [ eta = {:.2e}, etaorg = {:.2e}]".format(eta, etaorg))
        if (iterk % 3) == 0:
            normx = np.linalg.norm(x)
            normAtxi = np.linalg.norm(Atxi)
            normu = np.linalg.norm(u)
            if printyes:
                print(
                    "  [normx,Atxi,u ={:.2e}  {:.2e}  {:.2e} ]".format(
                        normx, normAtxi, normu
                    )
                )
        runhist.primobj.append(primobj)
        runhist.dualobj.append(dualobj)
        runhist.time.append(ttime)
        runhist.relgap.append(relgap)
        if breakyes > 0:
            # if printyes:
            #     print("{}".format(msg))
            break
        if primfeasorg < dualfeasorg:
            prim_win = prim_win + 1
        else:
            dual_win = dual_win + 1
        if iterk < 9:
            sigma_update_iter = 1
        elif iterk < 19:
            sigma_update_iter = 2
        elif iterk < 199:
            sigma_update_iter = 2
        elif iterk < 499:
            sigma_update_iter = 3
        sigmascale = 2
        sigmascale_reduce = 2**0.5
        sigmamax = 1e8
        if (iterk + 1) % sigma_update_iter == 0 or info_NCG.breakyes >= 0:
            sigmamin = 1e-4
            if prim_win > max(1, 1.2 * dual_win) and (info_NCG.breakyes < 0):
                prim_win = 0
                sigma = min(sigmamax, sigma * sigmascale)
            elif dual_win > max(1, 3 * prim_win) or info_NCG.breakyes >= 0:
                dual_win = 0
                sigma = max(sigmamin, sigma / sigmascale_reduce)
    # for循环结束
    """ recover orignal variables """
    if iterk == maxiter - 1:
        msg = " maximum iteration reached"
        info.termcode = 3
    ttime = time.time() - tstart

    xi = xi * np.sqrt(bscale * cscale)
    Atxi = ATmap0(xi)
    u = u * cscale
    x = x * bscale
    Ax = Ax * np.sqrt(bscale * cscale)
    Rd = Atxi + u
    Rp1 = Ax - borg
    normRp = np.linalg.norm(Rp1 - xi)
    normRp1 = np.linalg.norm(Rp1)
    normRd = np.linalg.norm(Rd)
    normu = np.linalg.norm(u)
    primfeasorg = normRp / normborg
    dualfeasorg = normRd / (1 + normu)

    primobj = 0.5 * np.linalg.norm(xi) ** 2 + lambda1org.T @ Sd(x)
    dualobj = -(0.5 * np.linalg.norm(xi) ** 2 + borg.T @ xi)
    primobjorg = 0.5 * normRp1**2 + lambda1org.T @ Sd(x)
    relgap = abs(primobj - dualobj) / max(1, abs(primobj))
    obj = [primobj, dualobj]
    grad = ATmap0(Rp1)
    etaorg = np.linalg.norm(x - proxSortedL1.proxSortedL1(x - grad, lambda1org)[0])
    eta = etaorg / (1 + np.linalg.norm(grad) + np.linalg.norm(x))
    gs = Sd(grad)
    gs_lambda1org = gs - lambda1org
    infeas = max(max(gs_lambda1org.cumsum()), 0) / lambda1org[0]

    runhist.m = m
    runhist.n = n
    ttime_cpu = time.perf_counter() - tstart_cpu
    runhist.iter = iterk
    runhist.totaltime = ttime
    runhist.primobjorg = primobj
    runhist.dualobjorg = dualobj
    runhist.maxfeas = max([dualfeasorg, primfeasorg])
    runhist.eta = eta
    runhist.etaorg = etaorg
    runhist.infeas = infeas
    info.infeas = infeas

    info.m = m
    info.n = n
    info.minx = min(min(x))
    info.maxx = max(max(x))
    info.relgap = relgap
    info.iterk = iterk
    info.time = ttime
    info.time_cpu = ttime_cpu
    info.admmtime = admm_admm.time
    info.admmtime_cpu = admm_admm.timecpu
    info.admmiter = admm_admm.iterk
    info.eta = eta
    info.etaorg = etaorg
    info.obj = obj
    info.dualfeasorg = dualfeasorg
    info.primfeasorg = primfeasorg
    info.maxfeas = max([dualfeasorg, primfeasorg])
    info.Axmb = normRp1
    info.nnzx = cardcal.cardcal(x, 0.999, 1e-16)
    info.x = x
    info.u = u
    info.xi = xi

    return obj, x, xi, u, info, runhist, monitor
