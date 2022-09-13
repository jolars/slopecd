"""
Created on Tue Jun 21 17:14:53 2022

@author: Prince_Li
"""

import numpy as np
import scipy.sparse


def mexfwsolve(LRt, r):
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
    if mr != LRt.shape[1]:
        print("R should be square!")
    if not scipy.sparse.issparse(LRt):
        print("R should be sparse!")
    Rt = LRt.data
    irRt = LRt.indices
    jcRt = LRt.indptr
    if irRt[jcRt[mr] - 1] < mr - 1:
        print(" R not lower triangular!")
    x = []
    x[mr - 1] = b[mr - 1] / Rt[jcRt[mr] - 1]
    for j in range(mr - 1, 0, -1):
        kstart = jcRt[j - 1] + 1
        kend = jcRt[j]
        tmp = 0
        for k in range(kstart, kend):
            idx = irRt[k]
            tmp += Rt[k] * x[idx]
        x[j - 1] = (b[j - 1] - tmp) / Rt[kstart - 1]
    if isspb:
        del b
    return x
