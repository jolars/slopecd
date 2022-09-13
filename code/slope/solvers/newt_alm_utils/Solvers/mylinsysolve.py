import numpy as np

from slope.solvers.newt_alm_utils.mexfun import mexbwsolve, mexfwsolve, mextriang


def mylinsysolve(L, r):
    qn = len(L.perm)
    q = np.zeros([qn, 1])
    if L.matfct_options == "cholesky":
        q[L.perm] = mextriang.mextriang(L.R, mextriang.mextriang(L.R, r[L.perm], 2), 1)
    elif L.matfct_options == "spcholmatlab":
        q[L.perm] = mexbwsolve.mexbwsolve(
            L.Rt, mexfwsolve.mexfwsolve(L.R, r(L.perm, 1))
        )
    return q
