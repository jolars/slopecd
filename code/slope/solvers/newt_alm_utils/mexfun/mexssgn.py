"""
Created on Tue Jun 21 09:56:59 2022

@author: Prince_Li
"""
import numpy as np
from numba import njit


@njit
def mexssgn(x):
    # x = np.sign(x) + (x == 0)
    # x[x == 0] = 1
    # return x
    return np.sign(x) + (x == 0).astype(np.float64)
