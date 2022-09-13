"""
%%**************************************************************
%% Jacobian = spdiags(hh,0,n,n) + U*U';
%%**************************************************************
"""


import numpy as np
from numba import njit

@njit
def WKJacobian(rr2):
    n = len(rr2)
    blklen = []
    blkend = []
    len_len = 0
    numblk = 0
    for k in range(len(rr2)):
        if rr2[k] == 1:
            len_len = len_len + 1
        else:
            if len_len > 0:
                numblk = numblk + 1
                blklen[numblk, 0] = len_len
                blkend[numblk, 0] = k
                len_len = 0
    if len_len > 0:
        numblk = numblk + 1
        blklen[numblk, 0] = len_len()
        blkend[numblk, 0] = n
    numblk = len(blklen)
    if numblk == 0:
        hh = np.ones([n, 1])
        U = []
    else:
        U = []
        hh = np.ones([n, 1])
    for k in range(numblk):
        if blkend(k) < n or len_len == 0:  #% k\neq N or N\notin J
            Len = blklen[k] + 1
            invsqrtlen = 1 / np.sqrt(Len)
            idxend = blkend[k]
            idxsub = range(idxend - blklen[k], idxend)
            hh[idxsub] = 0
            vv = np.zeros([n, 1])
            vv[idxsub] = invsqrtlen  # %%ones(len,1)/sqrt(len);
            U = np.stack((U, vv))
        else:
            idxsub = range(n - blklen[numblk] + 1, n)  # %% K=N and N\in J
            hh[idxsub] = 0
