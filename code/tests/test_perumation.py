#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 23:31:14 2022

@author: jonaswallin
"""

import unittest
import numpy as np
from scipy import sparse
from slope.permutation import permutation_matrix
import slope.permutation as slopep
from numpy.random import default_rng
from scipy.sparse.linalg import spsolve

#NOTE THAT THE INVERSE OF B.T is np.cumsum
class TestPerumation(unittest.TestCase):
    
    
    def test_permutation_matrix(self):
        x = np.array([1., 1., -2., 2., 3])
        Pi = permutation_matrix(x)
        Pix = Pi @ x 
        np.alltrue(Pix >= 0)
        np.alltrue(np.diff(Pix)>=0)
        
    def test_Projection(self):
        
        
        x = np.array([1., 1., -2., 2., 3])
        n  = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        Gamma = np.array([0,1,2])


class TestBfunctions(unittest.TestCase):
    
    def test_BtInv(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n  = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        BT = B.T
        np.testing.assert_array_almost_equal(x, slopep.BTinv(BT @ x))
        
    def test_BInv(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n  = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(x, slopep.Binv(B @ x))
        
    def test_B(self):
        
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n  = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(slopep.B(x), B @ x) 
        
    def test_BBT_inv_B(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n  = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(slopep.BBT_inv_B(x), spsolve(B @ B.T, B @ x ))

if __name__ == "__main__":
    unittest.main()
