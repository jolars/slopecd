"""
Created on Tue Jun 21 15:08:20 2022

@author: Prince_Li
"""

import numpy as np
import scipy.sparse


def mexfwsolve(LR, r):
    mr = r.shape[0]
    isspb = scipy.sparse.issparse(r)
    if isspb:
        btmp = r.data
        irb = r.indices  # 对应matlab 和 c 接口中的mxGetIr
        jcb = r.indptr
        b = np.zeros([mr, 1])
        kstart = jcb[0]
        kend = jcb[1]
        for k in range(kstart, kend):
            b[irb[k]] = btmp[k]
    else:
        b = r
    if mr != LR.shape[1]:
        print("R should be square!")
    if not scipy.sparse.issparse(LR):
        print("R should be sparse!")
    R = LR.data
    irR = LR.indices
    jcR = LR.indptr
    if irR[jcR[1] - 1] > 0:
        print(" R not upper triangular!")
    x = []
    x[0] = b[0] / R[0]
    for j in range(mr):
        kstart = jcR[j]
        kend = jcR[j + 1] - 1
        tmp = 0
        for k in range(kstart, kend):
            idx = irR[k]
            tmp += R[k] * x[idx]
        x[j] = (b[j] - tmp) / R[kend]
    if isspb:
        del b
    return x
