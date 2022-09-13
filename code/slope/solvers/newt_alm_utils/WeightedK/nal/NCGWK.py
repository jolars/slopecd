
import math

import numpy as np

from slope.solvers.newt_alm_utils.Solvers import proxSortedL1
from slope.solvers.newt_alm_utils.WeightedK.nal import WKsolve


def findstep(par, b, lambda1, Ly0, xi0, Atxi0, u0, up0, dxi, Atdxi, tol, options):
    eps = 2.2204e-16
    Atxi = 0
    if "stepop" in options.__dict__:
        stepop = options.stepop
    printlevel = 0
    maxit = math.ceil(math.log(1 / (tol + eps)) / math.log(2))
    c1 = 1e-4
    c2 = 0.9
    sig = par.sigma
    g0 = dxi.T @ (-b - xi0) + Atdxi.T @ up0
    if g0 <= 0:
        alp = 0
        iterk = 0
        if printlevel:
            print("\n Need an ascent direction, %2.1e  ", g0)
        xi = xi0
        Atxi = Atxi0
        u = u0
        up = up0
        Ly = Ly0
        return par, Ly, xi, Atxi, u, up, alp, iterk
    alp = 1
    alpconst = 0.5
    for iterk in range(maxit):
        if iterk == 0:
            alp = 1
            LB = 0
            UB = 1
        else:
            alp = alpconst * (LB + UB)
        xi = xi0 + alp * dxi
        uinput = up0 + sig * u0 - sig * alp * Atdxi
        [up, info_u] = proxSortedL1.proxSortedL1(uinput, sig * lambda1)
        par.info_u = info_u
        u = (uinput - up) / sig
        galp = dxi.T @ (-b - xi) + Atdxi.T @ up
        Ly = (
            -b.T @ xi
            - 0.5 * np.linalg.norm(xi) ** 2
            - np.linalg.norm(up) ** 2 / (2 * sig)
        )
        if printlevel:
            print("\n ------------------------------------- \n")
            print("\n alp = %4.3f, LQ = %11.10e, LQ0 = %11.10e", alp, Ly, Ly0)
            print("\n galp = %4.3f, g0 = %4.3f", galp, g0)
            print("\n ------------------------------------- \n")
        if iterk == 0:
            gLB = g0
            gUB = galp
            if np.sign(gLB) * np.sign(gUB) > 0:
                if printlevel:
                    print("|")
                Atxi = Atxi0 + alp * Atdxi
                return par, Ly, xi, Atxi, u, up, alp, iterk
        if ((abs(galp) < c2 * abs(g0))) and (
            Ly - Ly0 - c1 * alp * g0 > -1e-12 / max(1, abs(Ly0))
        ):
            if (stepop == 1) or ((stepop == 2) and (abs(galp) < tol)):
                if printlevel:
                    print(":")
                Atxi = Atxi0 + alp * Atdxi
                return par, Ly, xi, Atxi, u, up, alp, iterk
        if np.sign(galp) * np.sign(gUB) < 0:
            LB = alp
            gLB = galp
        elif np.sign(galp) * np.sign(gLB) < 0:
            UB = alp
            gUB = galp
    if iterk == maxit - 1:
        Atxi = Atxi0 + alp * Atdxi
    if printlevel:
        print("m")
    return par, Ly, xi, Atxi, u, up, alp, iterk


def NCGWK(b, Ainput, x0, Ax0, Atxi0, xi0, lambda1, par, options):
    eps = 2.2204e-16
    printsub = 0
    breakyes = 0
    maxitersub = 50
    tiny = 1e-10
    tol = 1e-6
    maxitpsqmr = 500
    use_proximal = 0

    if "printsub" in options.__dict__:
        printsub = options.printsub
    else:
        printsub = 0

    if "maxitersub" in options.__dict__:
        maxitersub = options.maxitersub
    else:
        maxitersub = 50

    if "tiny" in options.__dict__:
        tiny = options.tiny
    else:
        tiny = 1e-10

    if "tol" in options.__dict__:
        tol = min(tol, options.tol)
    else:
        tol = 1e-6

    if "use_proximal" in options.__dict__:
        use_proximal = options.use_proximal
    else:
        use_proximal = 0
    sig = par.sigma
    bscale = options.bscale
    cscale = options.cscale
    normborg = 1 + np.linalg.norm(b) * np.sqrt(bscale * cscale)

    """ preperation """
    Amap_ncg = lambda x: Ainput.A_submit_A @ x * Ainput.bs_con_int
    ATmap_ncg = lambda x: Ainput.A_submit_A.T @ x * Ainput.bs_con_int
    par.lsAmap_ncg = Amap_ncg
    uinput = x0 - sig * Atxi0
    up, info_u = proxSortedL1.proxSortedL1(uinput, sig * lambda1)
    par.info_u = info_u
    u = (uinput - up) / sig
    Rp = Ax0 - b - xi0
    normRp = np.linalg.norm(Rp)
    Atxi = Atxi0
    xi = xi0
    Ly = -b.T @ xi - 0.5 * np.linalg.norm(xi) ** 2 - np.linalg.norm(up) ** 2 / (2 * sig)

    class runhist:
        pass

    runhist.psqmr = []
    runhist.findstep = []
    runhist.priminf = []
    runhist.dualinf = []
    runhist.Ly = []
    runhist.solve_ok = []
    runhist.psqmr.append(0)
    runhist.findstep.append(0)

    """ main Newton iteration """
    for itersub in range(maxitersub):
        xiold = xi
        Atxiold = Atxi
        Rd = Atxi + u
        normRd = np.linalg.norm(Rd)
        Aup = Amap_ncg(up) * Ainput.sbc_submit
        GradLxi = -(xi + b - Aup)
        normGradLxi = np.linalg.norm(GradLxi) * np.sqrt(bscale * cscale) / normborg
        priminf_sub = normGradLxi
        dualinf_sub = normRd * cscale / (1 + np.linalg.norm(u) * cscale)
        if max(priminf_sub, dualinf_sub) < tol:
            tolsubconst = 0.1
        else:
            tolsubconst = 0.05
        tolsub = max(min(1, par.tolconst * dualinf_sub), tolsubconst * tol)
        runhist.priminf.append(priminf_sub)
        runhist.dualinf.append(dualinf_sub)
        runhist.Ly.append(Ly)
        if printsub:
            print(
                "  {}  {:.5e}  {:.2e}  {:.2e} {:.2e}".format(
                    itersub, Ly[0][0], priminf_sub, dualinf_sub, par.tolconst
                ),
                end=" ",
            )
        if normGradLxi < tolsub and itersub > 0:
            msg = "good termination in subproblem:"
            if printsub:
                print(
                    "\n {} normRd={:.2e}, gradLyxi = {:.2e}, tolsub={:.2e}".format(
                        msg, normRd, normGradLxi, tolsub
                    )
                )
            breakyes = -1
            break
        """ Compute Newton direction """
        par.epsilon = min([1e-3, 0.1 * normGradLxi])  # good to add
        if (dualinf_sub > 1e-3) or (itersub <= 4):
            maxitpsqmr = max(maxitpsqmr, 200)
        elif dualinf_sub > 1e-4:
            maxitpsqmr = max(maxitpsqmr, 300)
        elif dualinf_sub > 1e-5:
            maxitpsqmr = max(maxitpsqmr, 400)
        elif dualinf_sub > 5e-6:
            maxitpsqmr = max(maxitpsqmr, 500)
        if itersub > 0:
            prim_ratio = priminf_sub / runhist.priminf[itersub - 0]
            dual_ratio = dualinf_sub / runhist.dualinf[itersub - 0]
        else:
            prim_ratio = 0
            dual_ratio = 0
        rhs = GradLxi
        tolpsqmr = min([5e-3, 0.1 * np.linalg.norm(rhs)])
        const2 = 1
        if itersub > 0 and (prim_ratio > 0.5 or priminf_sub > 0.1 * runhist.priminf[0]):
            const2 = 0.5 * const2
        if dual_ratio > 1.1:
            const2 = 0.5 * const2
        tolpsqmr = const2 * tolpsqmr
        par.tol = tolpsqmr
        par.maxit = 2 * maxitpsqmr
        [dxi, resnrm, solve_ok, par] = WKsolve.WKsolve(Ainput, rhs, par)
        Atdxi = ATmap_ncg(dxi) * Ainput.sbc_submit
        if isinstance(resnrm, int):
            iterpsqmr = 0
        else:
            iterpsqmr = len(resnrm) - 1
        if printsub:
            if isinstance(resnrm, int):
                print(
                    "| {:.1e}  {:.1e} {} {:.1f}  [{}, {}, ({}, {})] ".format(
                        par.tol,
                        resnrm,
                        iterpsqmr,
                        const2,
                        int(sum(par.info_u.rr2)[0]),
                        int(par.info_u.nz),
                        par.lenP,
                        par.numblk1,
                    ),
                    end=" ",
                )
            else:
                print(
                    "| {:.1e}  {:.1e} {} {:.1f}  [{}, {}, ({}, {})]".format(
                        par.tol,
                        resnrm[-1],
                        iterpsqmr,
                        const2,
                        int(sum(par.info_u.rr2)[0]),
                        int(par.info_u.nz),
                        par.lenP,
                        par.numblk1,
                    ),
                    end=" ",
                )

        par.iter = itersub
        if (itersub <= 2) and (dualinf_sub > 1e-4) or (par.iter < 2):
            stepop = 1
        else:
            stepop = 2

        class step_op:
            pass

        steptol = 1e-5
        step_op.stepop = stepop
        par, Ly, xi, Atxi, u, up, alp, iterstep = findstep(
            par, b, lambda1, Ly, xi, Atxi, u, up, dxi, Atdxi, steptol, step_op
        )
        runhist.solve_ok.append(solve_ok)
        runhist.psqmr.append(iterpsqmr)
        runhist.findstep.append(iterstep)
        Ly_ratio = 1
        if itersub > 0:
            Ly_ratio = (Ly - runhist.Ly[itersub - 1]) / (abs(Ly) + eps)
        if printsub:
            print(" {:.2e} {:.0f}".format(alp, iterstep))
            if Ly_ratio < 0:
                print("-")
        """ check for stagnation """
        if itersub > 3:
            idx = range(max(1, itersub - 3), itersub + 1)
            tmp = np.array(runhist.priminf)[idx]
            ratio = min(tmp) / max(tmp)
            if (
                (all(np.array(runhist.solve_ok)[idx] <= -1))
                and (ratio > 0.9)
                and (min(runhist.psqmr(idx)) == max(runhist.psqmr(idx)))
                and (max(tmp) < 5 * tol)
            ):
                print("#")
                breakyes = 1
            const3 = 0.7
            priminf_1half = min(
                np.array(runhist.priminf)[range(0, math.ceil(itersub * const3) + 1)]
            )
            priminf_2half = min(
                np.array(runhist.priminf)[
                    range(math.ceil(itersub * const3) + 1, itersub + 1)
                ]
            )
            priminf_best = min(np.array(runhist.priminf)[range(0, itersub - 1)])
            priminf_ratio = runhist.priminf[itersub] / runhist.priminf[itersub - 1]
            dualinf_ratio = runhist.dualinf[itersub] / runhist.dualinf[itersub - 1]
            solve_ok_1_itersub = np.array(runhist.solve_ok)[range(0, itersub + 1)]
            stagnate_idx = [
                i for (i, val) in enumerate(solve_ok_1_itersub) if val <= -1
            ]
            stagnate_count = len(stagnate_idx)
            idx2 = range(max(0, itersub - 7), itersub + 1)
            if (
                (itersub >= 9)
                and all(np.array(runhist.solve_ok)[idx2] == -1)
                and (priminf_best < 1e-2)
                and (dualinf_sub < 1e-3)
            ):
                tmp = runhist.priminf[idx2]
                ratio = min(tmp) / max(tmp)
                if ratio > 0.5:
                    if printsub:
                        print("##")
                    breakyes = 2
            if (
                (itersub >= 14)
                and (priminf_1half < min(2e-3, priminf_2half))
                and (dualinf_sub < 0.8 * runhist.dualinf[0])
                and (dualinf_sub < 1e-3)
                and (stagnate_count >= 3)
            ):
                if printsub:
                    print("###")
                breakyes = 3
            if (
                (itersub >= 14)
                and (priminf_ratio < 0.1)
                and (priminf_sub < 0.8 * priminf_1half)
                and (dualinf_sub < min(1e-3, 2 * priminf_sub))
                and (
                    (priminf_sub < 2e-3) or (dualinf_sub < 1e-5 and priminf_sub < 5e-3)
                )
                and (stagnate_count >= 3)
            ):
                if printsub:
                    print("$$")
                breakyes = 4
            if (
                (itersub >= 9)
                and (dualinf_sub > 5 * min(runhist.dualinf))
                and (priminf_sub > 2 * min(runhist.priminf))
            ):
                if printsub:
                    print("$$$")
                breakyes = 5
            if itersub >= 19:
                dualinf_ratioall = (
                    np.array(runhist.dualinf)[range(1, itersub)]
                    / np.array(runhist.dualinf)[range(0, itersub - 1)]
                )
                idx = [i for (i, val) in enumerate(dualinf_ratioall) if val > 1]
                if len(idx) >= 3:
                    dualinf_increment = np.mean(np.array(dualinf_ratioall)[idx])
                    if dualinf_increment > 1.25:
                        if printsub:
                            print("^^")
                        breakyes = 6
            if breakyes > 0:
                Rd = Atxi + u
                normRd = np.linalg.norm(Rd)
                Aup = Amap_ncg(up) * Ainput.sbc_submit
                print(
                    "\n new dualfeasorg = {:.2e}".format(
                        normRd * cscale / (1 + np.linalg.norm(u) * cscale)
                    )
                )

    class info:
        pass

    info.breakyes = breakyes
    info.itersub = itersub
    info.tolconst = par.tolconst
    info.up = up
    info.Aup = Aup
    return u, Atxi, xi, par, runhist, info
