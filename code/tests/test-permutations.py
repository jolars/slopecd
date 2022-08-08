#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 23:31:14 2022

@author: jonaswallin
"""

import unittest

import numpy as np
import numpy.random as npr
from numpy.random import default_rng
from scipy import sparse
from scipy.sparse.linalg import spsolve

import slope.permutation as slopep
from slope.permutation import permutation_matrix

# NOTE THAT THE INVERSE OF B.T is np.cumsum


class TestPerumation(unittest.TestCase):
    def test_permutation_matrix(self):
        x = np.array([1.0, 1.0, -2.0, 2.0, 3])
        Pi = permutation_matrix(x)
        Pix = Pi @ x

        self.assertTrue(np.alltrue(Pix >= 0))
        self.assertTrue(np.alltrue(np.diff(Pix) <= 0))


class TestBfunctions(unittest.TestCase):
    def test_BtInv(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        BT = B.T
        np.testing.assert_array_almost_equal(x, slopep.BTinv(BT @ x))

    def test_BInv(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(x, slopep.Binv(B @ x))

    def test_B(self):

        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(slopep.B(x), B @ x)

    def test_BBT_inv_B(self):
        n = 10
        rng = default_rng(9)
        x = rng.standard_normal(n)
        n = x.shape[0]
        B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")
        np.testing.assert_array_almost_equal(
            slopep.BBT_inv_B(x), spsolve(B @ B.T, B @ x)
        )

    def test_pi(self):

        xs = [
            np.array([3.0, 3.0, 0.0, 1.0, -1, 2, -3]),
            np.array([3.0, 0.0, 0.0, 1.0, -1, 2, -3]),
            npr.randint(-5, 5, 10),
        ]
        for x in xs:
            pi = slopep.permutation_matrix(x)
            pi_list, piT_list = slopep.build_pi(x)
            np.testing.assert_array_almost_equal(pi @ x, slopep.pix(x, pi_list))
            np.testing.assert_array_almost_equal(pi.T @ x, slopep.pix(x, piT_list))

    def test_build_AMAT(self):

        xs = [
            np.array([3.0, 3.0, 0.0, 1.0, -1, 2, -3]),
            np.array([3.0, 0.0, 0.0, 1.0, -1, 2, -3]),
            npr.randint(-5, 5, 10),
        ]
        for x in xs:
            m = 10
            # more test larger zero space, etc

            n = x.shape[0]
            rng = default_rng(9)
            A = rng.standard_normal((m, n))
            ##
            # method 1
            ##
            B = sparse.eye(n, format="csc") - sparse.eye(n, k=1, format="csc")

            pi = slopep.permutation_matrix(x)
            x_ord = pi @ x
            Gamma = np.where(B @ x_ord == 0)[0]
            B_Gamma = B[Gamma, :]

            P = sparse.eye(n, format="csc") - B_Gamma.T @ spsolve(
                B_Gamma @ B_Gamma.T, B_Gamma
            )
            M = pi.T @ P @ pi

            V = A @ M @ A.T

            ##
            # method 2
            ##
            GammaC = np.setdiff1d(np.arange(x.shape[0]), Gamma)
            start = 0
            nC = GammaC.shape[0]
            VW = np.zeros((m, nC))
            for i in range(nC):
                ind = np.arange(start, GammaC[i] + 1)
                VW[:, i] += np.sum(A @ pi[ind, :].T, 1)
                if ind.shape[0] > 1:
                    VW[:, i] /= np.sqrt(ind.shape[0])
                start = GammaC[i] + 1

            ##
            # method 3
            ##

            start = 0
            nC = GammaC.shape[0]
            VW = np.zeros((m, nC))
            pi_list, piT_list = slopep.build_pi(x)
            for i in range(nC):
                ind = np.arange(start, GammaC[i] + 1)
                for j in ind:
                    VW[:, i] += pi_list[j, 1] * A[:, pi_list[j, 0]]
                if ind.shape[0] > 1:
                    VW[:, i] /= np.sqrt(ind.shape[0])
                start = GammaC[i] + 1

            np.testing.assert_array_almost_equal(V, VW @ VW.T)

            np.testing.assert_array_almost_equal(V, VW @ VW.T)


if __name__ == "__main__":
    x = np.array([3.0, 3.0, 0.0, 1.0, -1, 2, -3])
    pi_list, piT_list = slopep.build_pi(x)
    pi = slopep.permutation_matrix(x)
    unittest.main()
