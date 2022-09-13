import numpy as np
import scipy.sparse

from slope.solvers.newt_alm_utils.mexfun import mexWK
from slope.solvers.newt_alm_utils.Solvers import mychol, mylinsysolve, psqmry
from slope.solvers.newt_alm_utils.WeightedK.nal import WKJacobian


def intersect(a, b):
    blist = []
    blistidx = []
    for i in a:
        if i in b:
            blist.append(i)
            blistidx.append(b.index(i))
    return blist, blistidx


def WKsolve(Ainput, rhs, par):
    options = 2
    resnrm = 0
    solve_ok = 1
    rr2 = par.info_u.rr2
    m = len(rhs)
    if False:
        h, U = WKJacobian.WKJacobian(rr2)
    else:
        h, U = mexWK.mexWK(rr2)
    op = 2
    oopp = 0
    if oopp == 1:
        Idnn = np.eye(len(par.info_u.rr2))
        if not U:
            VV2 = Ainput.A @ np.diag(par.info_u.s) @ Idnn[par.info_u.idx, :].T @ U
        else:
            VV2 = []
        VV1 = Ainput.A @ np.diag(par.info_u.s) @ Idnn[par.info_u.idx, :].T @ np.diag(h)
    vidx = [i for (i, val) in enumerate(h) if val > 0]
    v1idx = list(np.array(par.info_u.idx)[vidx])
    V1 = Ainput.A[:, v1idx]

    if U.size != 0:
        uu = np.sum(U, axis=1)
        idx_U = [i for (i, val) in enumerate(uu) if val > 0]
        Unew = U[idx_U, :]
        iidex = [i for (i, val) in enumerate(par.info_u.s) if val < 0]
        idx_A = list(np.array(par.info_u.idx)[idx_U])
        if op == 1:
            sgnidx = set(iidex).intersection(idx_A)
            tmp1 = -Ainput.A[:, sgnidx]
            Ainput.A[:, sgnidx] = tmp1
            Ahat = Ainput.A[:, idx_A]
            V2 = Ahat @ Unew
        else:
            Ahat = Ainput.A[:, idx_A]
            _, idA = intersect(iidex, idx_A)
            Ahat[:, idA] = -Ahat[:, idA]
            V2 = Ahat @ Unew
    else:
        V2 = []
    par.lenP = V1.shape[1]
    if len(V2) == 0:
        par.numblk1 = 0
    else:
        par.numblk1 = V2.shape[1]
    if (par.lenP + par.numblk1 > 6e3 and m > 6e3) or (
        par.lenP + par.numblk1 > 1e4 and m > 1000
    ):
        options = 1
    if options == 1:
        par.V1 = V1
        par.V2 = V2
        par.precond = 0
        xi, _, resnrm, solve_ok = psqmry.psqmry("mvWK", Ainput, rhs, par)
    else:
        if m < (par.lenP + par.numblk1):
            if len(V2) != 0:
                tmpM1 = par.sigma * (V1 @ V1.T + V2 @ V2.T)
            else:
                tmpM1 = par.sigma * (V1 @ V1.T)
            Meye = np.eye(m)
            M = scipy.sparse.csc_matrix(Meye) + tmpM1
            L = mychol.mychol(M, len(M))
            xi = mylinsysolve.mylinsysolve(L, rhs)
        else:
            if len(V2) == 0:
                W = V1
                nW = W.shape[1]
                SMWmat = W.T @ W
                Meye = np.eye(nW) / par.sigma
                SMWmat = scipy.sparse.csc_matrix(Meye) + SMWmat
                L = mychol.mychol(SMWmat, nW)
                xi = rhs - W @ mylinsysolve.mylinsysolve(L, (rhs.T @ W).T)
            else:
                W = np.hstack((V1, V2))
                nW = W.shape[1]
                SMWmat = W.T @ W
                Meye = np.eye(nW) / par.sigma
                SMWmat = scipy.sparse.csc_matrix(Meye) + SMWmat
                L = mychol.mychol(SMWmat, nW)
                xi = rhs - W @ mylinsysolve.mylinsysolve(L, (rhs.T @ W).T)
    par.innerop = options
    return xi, resnrm, solve_ok, par
