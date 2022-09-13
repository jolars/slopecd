"""
Created on Thu Jun 16 16:33:43 2022

@author: Prince_Li
"""

"""
/***********************************************************************
* mexMatvec: compute
* 
* mexMatvec(A,y,options)
*          
*  options = 0, compute A*y
*          = 1, compute (y'*A)' 
*
* Copyright (c) 2004 by
* K.C. Toh
* Last Modified: 120404   
************************************************************************/
"""

from typing import Dict  # 判断变量类型是否为Dict

import numpy as np
import scipy.sparse
from numba import njit


def mexMatvec(AA, xx, rr=0):
    if isinstance(AA, Dict) or isinstance(xx, Dict):
        print("A, x must be a double array")
    # /***** assign pointers *****/
    # A = mxGetPr(prhs[0])
    # m1 = mxGetM(prhs[0])
    # n1 = mxGetN(prhs[0])
    A = AA
    m1, n1 = A.shape
    isspA = scipy.sparse.issparse(A)
    if isspA:
        irA = A.indices
        jcA = A.indptr
    isspy = scipy.sparse.issparse(xx)
    m2, n2 = xx.shape
    if n2 > 1:
        print("2ND input must be a column vector")
    if isspy:
        iry = xx.indices
        jcy = xx.indptr
        ytmp = xx.data
        # /***** copy ytmp to y *****/
        y = np.zeros([m2, 1])
        kstart = jcy[0]
        kend = jcy[1]
        for k in range(kstart, kend):
            r = iry[k]
            y[r] = ytmp[k]
    else:
        y = xx

    if rr == 0 and n1 != m2:
        print("1ST and 2ND input not compatible.")
    elif rr and m1 != m2:
        print("1ST and 2ND input not compatible.")
    # /***** create return argument *****/
    if rr == 0:
        Ay = np.zeros([m1, 1])
    else:
        Ay = np.zeros([n1, 1])

    # /***** main body *****/
    if rr == 0:
        if not isspA:
            for j in range(n1):
                jm1 = j * m1
                tmp = y[j]
                if tmp != 0:
                    Ay = saxpymat(A, jm1, 0, m1, tmp, Ay, 0)
        else:
            for j in range(n1):
                tmp = y[j]
                if tmp != 0:
                    istart = jcA[j]
                    iend = jcA[j + 1]
                    for i in range(istart, iend):
                        r = irA[i]
                        Ay[r] += tmp * A[i]
    else:
        if not isspA:
            for j in range(n1):
                jm1 = j * m1
                # print(jm1)
                Ay[j] = realdotde(A, jm1, y, m1)
        else:
            for j in range(n1):
                istart = jcA[j]
                iend = jcA[j + 1]
                tmp = 0
                for i in range(istart, iend):
                    r = irA[i]
                    tmp += y[r] * A[i]
                Ay[j] = tmp

    return Ay


"""
/********************************
* realdotde: x dense matrix,  
*            y dense vector
*********************************/
"""


@njit
def realdotde(x, idx, y, n):
    mm, nn = x.shape
    x = x.reshape(mm * nn, 1)
    r = 0

    for i in range(n - 3):
        # print(x[i+idx])
        # LEVEL 4
        r += x[i + idx] * y[i]
        i += 1
        r += x[i + idx] * y[i]
        i += 1
        r += x[i + idx] * y[i]
        i += 1
        r += x[i + idx] * y[i]
    # LEVEL 2
    if i < n - 1:
        r += x[i + idx] * y[i]
        i += 1
        r += x[i + idx] * y[i]
        i += 1
    # LEVEL 1
    if i < n - 1:
        r += x[i + idx] * y[i]
    return r


"""
/********************************
* saxpymat:  z = z + alpha*y 
* y dense matrix, z dense vector
********************************/
"""


@njit
def saxpymat(y, idx1, istart, iend, alp, z, idx2):
    for i in range(istart, iend - 3):
        # LEVEL 4
        z[i + idx2] += alp * y[i + idx1]
        i += 1
        z[i + idx2] += alp * y[i + idx1]
        i += 1
        z[i + idx2] += alp * y[i + idx1]
        i += 1
        z[i + idx2] += alp * y[i + idx1]
    # LEVEL 2
    if i < iend - 1:
        z[i + idx2] += alp * y[i + idx1]
        i += 1
        z[i + idx2] += alp * y[i + idx1]
        i += 1
    if i < iend:
        z[i + idx2] += alp * y[i + idx1]
    return z


"""/**********************************************************/"""
# 这里有个mxArray不懂怎么使用  指针指的是数组的第一个元素
# def mexFunction(nlhs,plhs,nrhs,prhs):

#    if len(args) < 2:
#        print('mexMatvec: must have at least 2 inputs')
# /* CHECK THE DIMENSIONS */
# nlhs：输出参数数目
# plhs：指向输出参数的指针
# nrhs：输入参数数目
# prhs：输入参数
# matlab中判断是否为cell类型, 对应python中的dict 和matlab中的cell 结构相似
# ---------------------------------------------------------------------#
# if (mxIsCell(prhs[0]) or mxIsCell(prhs[1])):
#     mexErrMsgTxt(" mexMatvec: A, x must be a double array")
# if nrhs <2:  # 输入参数数目是否小于2
#     mexErrMsgTxt(" mexMatvec: must have at least 2 inputs")
# if nlhs > 2: # 输出参数数目是否大于2 python 不需要判断
#     mexErrMsgTxt("mexMatvec: requires 1 output argument")
# if nrhs == 2: # 输入参数数目等于2
#     options = 0
# else:
#     options = mxGetPr(prhs[2])
# ---------------------------------------------------------------------#
