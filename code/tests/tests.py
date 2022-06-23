import unittest
from bisect import bisect_right

import numpy as np
import scipy.sparse as sparse
from benchopt.datasets.simulated import make_correlated_data

from slope.clusters import get_clusters, update_cluster
from slope.solvers import admm, hybrid_cd, newt_alm, prox_grad
from slope.utils import lambda_sequence


class TestHybridSolver(unittest.TestCase):
    def test_convergence(self):
        X, y, _ = make_correlated_data(n_samples=100, n_features=30, random_state=0)

        tol = 1e-6
        q = 0.4
        reg = 0.01

        for X_sparse in [False, True]:
            if X_sparse:
                X = sparse.csc_matrix(X)
            for fit_intercept in [False, True]:
                lambdas = lambda_sequence(X, y, fit_intercept, reg=reg, q=q)

                _, _, _, gaps, _ = hybrid_cd(
                    X, y, lambdas, fit_intercept=fit_intercept, tol=tol
                )

                with self.subTest():
                    self.assertGreater(tol, gaps[-1])


class TestNewtALMSolver(unittest.TestCase):
    def test_convergence(self):
        X, y, _ = make_correlated_data(n_samples=20, n_features=200, random_state=0)

        tol = 1e-6
        q = 0.5
        reg = 0.02

        for X_sparse in [False, True]:
            if X_sparse:
                X = sparse.csc_matrix(X)
            for fista in [False, True]:
                for fit_intercept in [False, True]:
                    lambdas = lambda_sequence(X, y, fit_intercept, reg=reg, q=q)

                    _, _, _, gaps, _ = newt_alm(X, y, lambdas, tol=tol)

                    with self.subTest():
                        self.assertGreater(tol, gaps[-1])


class TestPGDSolvers(unittest.TestCase):
    def test_convergence(self):
        X, y, _ = make_correlated_data(n_samples=30, n_features=70, random_state=0)

        tol = 1e-6
        q = 0.5
        reg = 0.02

        for X_sparse in [False, True]:
            if X_sparse:
                X = sparse.csc_matrix(X)
            for fista in [False, True]:
                for fit_intercept in [False, True]:
                    lambdas = lambda_sequence(X, y, fit_intercept, reg=reg, q=q)

                    _, _, _, gaps, _ = prox_grad(
                        X, y, lambdas, fista=fista, fit_intercept=fit_intercept, tol=tol
                    )

                    with self.subTest():
                        self.assertGreater(tol, gaps[-1])


class TestADMMSolver(unittest.TestCase):
    def test_admm_convergence(self):
        for p in [50, 150]:
            X, y, _ = make_correlated_data(n_samples=100, n_features=p, random_state=51)

            reg = 0.1
            q = 0.3
            tol = 1e-5

            for X_sparse in [False, True]:
                if X_sparse:
                    X = sparse.csc_matrix(X)

                for fit_intercept in [False, True]:
                    lambdas = lambda_sequence(X, y, fit_intercept, reg=reg, q=q)

                    _, _, _, gaps, _ = admm(
                        X, y, lambdas, fit_intercept=fit_intercept, tol=tol
                    )

                    with self.subTest():
                        self.assertGreater(tol, gaps[-1])


class TestClusterUpdates(unittest.TestCase):
    def test_cluster_updates(self):

        for i in range(1000):
            seed = i
            rng = np.random.default_rng(seed)

            beta = rng.integers(low=1, high=5, size=8).astype(float)

            # beta = np.array([2.0, 1.5, 2.0, 0.2])

            c, c_ptr, c_ind, n_c = get_clusters(beta)

            ind_old = rng.integers(0, n_c)

            new_coef = rng.integers(low=0, high=6).astype(float)

            ind_new = n_c - bisect_right(c[:n_c][::-1], abs(new_coef))

            if ind_new > ind_old and c[ind_new] != abs(new_coef):
                ind_new -= 1

            cluster = c_ind[c_ptr[ind_old] : c_ptr[ind_old + 1]].copy()
            new_beta = beta.copy()
            new_beta[cluster] = new_coef

            n_c = update_cluster(c, c_ptr, c_ind, n_c, new_coef, ind_old, ind_new)

            c_true, c_ptr_true, c_ind_true, n_c_true = get_clusters(new_beta)

            self.assertEqual(n_c, n_c_true)
            np.testing.assert_array_equal(c_true[:n_c_true], c[:n_c])
            np.testing.assert_array_equal(c_ptr[: n_c + 1], c_ptr_true[: n_c_true + 1])

            for i in range(n_c):
                a = np.sort(c_ind[c_ptr[i] : c_ptr[i + 1]])
                b = np.sort(c_ind_true[c_ptr_true[i] : c_ptr_true[i + 1]])
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
