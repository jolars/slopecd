"""
Created on Tue Jun 21 20:20:52 2022

@author: Prince_Li
"""


import numpy as np
import scipy.sparse


def mexWK(x):
    nr = len(x)
    if scipy.sparse.issparse(x):
        print("input cannot be sparse.")
    else:
        rr = x
    if nr == 0:
        print("input size cannot be zero")
    blklen = np.zeros([nr, 1])
    blkend = np.zeros([nr, 1])
    leng = 0
    numblk = 0
    NZ = 0
    endflag = 0
    for k in range(nr):
        if rr[k] == 1:
            leng += 1
        else:
            if leng > 0:
                blklen[numblk] = leng
                blkend[numblk] = k
                NZ = NZ + leng + 1
                numblk += 1
                leng = 0
    if leng > 0:
        blklen[numblk] = leng
        blkend[numblk] = nr
        endflag = 1
        numblk += 1
    h = np.zeros([nr, 1])
    if endflag == 1:
        Us = np.zeros([nr, numblk - 1])
        U = scipy.sparse.csc_matrix(Us)
        del Us
    else:
        Us = np.zeros([nr, numblk])
        U = scipy.sparse.csc_matrix(Us)
        del Us
    hh = h
    ii = list(U.indices)
    jj = U.indptr
    vv = list(U.data)
    cnt = 0
    if numblk > 0:
        for k in range(numblk):
            idxend = blkend[k]
            if idxend == nr:
                leng = blklen[k]
                idxstart = idxend - leng
                for j in range(int(idxstart[0]), int(idxend[0])):
                    hh[j] = 1
            else:
                leng = blklen[k]
                idxstart = idxend - leng
                tmp = 1 / np.sqrt(leng + 1)
                for j in range(int(idxstart[0]), int(idxend[0]) + 1):
                    hh[j] = 1
                    ii.append(j)
                    vv.append(tmp)
                    # ii.append(j)
                    # vv.append(tmp)
                    cnt += 1
                jj[k + 1] = cnt
        for k in range(nr):
            hh[k] = 1 - hh[k]
    nii = len(ii)
    ii = np.array(ii).reshape(
        nii,
    )
    nvv = len(vv)
    vv = np.array(vv).reshape(
        nvv,
    )
    njj = len(jj)
    jj = np.array(jj).reshape(
        njj,
    )
    if endflag == 1:
        U = scipy.sparse.csc_matrix((vv, ii, jj), shape=(nr, numblk - 1))
    else:
        U = scipy.sparse.csc_matrix((vv, ii, jj), shape=(nr, numblk))
    return hh, U
