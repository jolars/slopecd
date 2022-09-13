"""
Created on Tue Jun 21 19:03:10 2022

@author: Prince_Li
"""


import numpy as np
import scipy.sparse
from numba import njit

"""
/*************************************************************
   TIME-CRITICAL PROCEDURE -- subscalarmul(x,alpha,y,n)
   Computes x -= alpha * y using LEVEL 8 loop-unrolling.
**************************************************************/
"""


@njit
def subscalarmul(x, alpha, y, n):
    """/* LEVEL 8 */"""
    for ii in range(n - 7):
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
    if ii < n - 3:
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
    if ii < n - 1:
        x[ii] -= alpha * y[ii]
        ii = ii + 1
        x[ii] -= alpha * y[ii]
        ii = ii + 1
    if ii < n:
        x[ii] -= alpha * y[ii]
    return x


# '''
# PROCEDURE ubsolve -- Solves xnew from U * xnew = x,
# where U is upper-triangular.
# INPUT
# u,n - n x n full matrix with only upper-triangular entries
# UPDATED
# x - length n vector
# On input, contains the right-hand-side.
# On output, xnew = U\xold
# '''


@njit
def ubsolve(x, u, n):
    SQR = lambda y: y * y
    """
      /*------------------------------------------------------------
     At each step j= n-1,...0, we have a (j+1) x (j+1) upper triangular
     system "U*xnew = x". The last equation gives:
       xnew[j] = x[j] / u[j,j]
     Then, we update the right-hand side:
       xnew(0:j-1) = x(0:j-1) - xnew[j] * u(0:j-1)
   --------------------------------------------------------------*/
    """
    j = n
    u = u + SQR(n)
    while j > 0:
        j = j - 1
        u = u - n
        x[j] = x[j] / u[j]
        x = subscalarmul(x, x[j], u, j)
    return x


@njit
def realdot(x, y, nn):
    r = 0
    i_list = []
    if nn != 0:
        for ii in range(nn - 7):
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
            ii = ii + 1
            i_list.append(ii)
            r += x[ii] * y[ii]
        if len(i_list) != 0:
            if i_list[-1] < nn - 3:
                ii = i_list[-1]
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
        if len(i_list) != 0:
            if i_list[-1] < nn - 1:
                ii = i_list[-1]
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
                r += x[ii] * y[ii]
                ii = ii + 1
                i_list.append(ii)
        if len(i_list) != 0:
            if i_list[-1] < nn:
                ii = i_list[-1]
                r += x[ii] * y[ii]
    return r


# '''
# /*************************************************************
#    PROCEDURE lbsolve -- Solve y from U'*y = x.
#    INPUT
#      u - n x n full matrix with only upper-triangular entries
#      x,n - length n vector.
#    OUTPUT
#      y - length n vector, y = U'\x.
# **************************************************************/
# '''


def lbsolve(y, u, x, n):
    """
     /*------------------------------------------------------------
     The first equation, u(:,1)'*y=x(1), yields y(1) = x(1)/u(1,1).
     For k=2:n, we solve
        u(1:k-1,k)'*y(1:k-1) + u(k,k)*y(k) = x(k).
    -------------------------------------------------------------*/
    """
    for k in range(n):
        u = u + n
        y[k, 0] = (x[k, 0] - realdot(y, u, k)) / u[k, 0]
    return y


def mextriang(LR, r, rr):
    U = LR
    if scipy.sparse.issparse(LR):
        print("Sparse U not supported.")
    n = LR.shape[0]
    y = np.zeros([n, 1])
    if LR.shape[1] != n:
        print("U should be square and upper triangular.")
    isspb = scipy.sparse.issparse(r)
    if r.shape[0] * r.shape[1] != n:
        print("size of U,b mismatch")
    if isspb:
        btmp = r.data
        irb = r.indices
        jcb = r.indptr
        b = np.zeros([n, 1])
        kend = jcb[1]
        for k in range(kend):
            b[irb[k]] = btmp[k]
    else:
        b = r
    if rr == 1:
        # for k in range(n):
        #     y[k] = b[k]
        y = np.linalg.inv(U) @ b  # bsolve(y,U,n)
    elif rr == 2:
        y = np.linalg.inv(U.T) @ b  # lbsolve(y,U,b,n)
    if isspb:
        del b
    return y
