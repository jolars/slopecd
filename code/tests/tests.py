import unittest
from parameterized import parameterized

import numpy as np
from scipy import stats
from benchopt.datasets.simulated import make_correlated_data

import slope
from slope.utils import dual_norm_slope
from slope.solvers import prox_grad


class TestPenalty(unittest.TestCase):
    def test_results(self):
        beta = np.array([3, 2.5, 1.2, -4, 0, -0.2])
        lam = np.array([2.7, 2, 1.7, 1.1, 0.8, 0.4])

        pen = slope.SortedL1Norm(lam)

        self.assertAlmostEqual(pen.evaluate(beta), 22.53)

        np.testing.assert_allclose(pen.prox(beta),
                                   np.array([1.0, 0.8, 0.1, -1.3, 0.0, -0.0]))

    def test_assertions(self):
        with self.assertRaises(ValueError):
            slope.SortedL1Norm(np.array([0.1, 0.2]))
        with self.assertRaises(ValueError):
            slope.SortedL1Norm(np.array([0.2, -0.2]))


class TestPGDSolvers(unittest.TestCase):
    def test_convergence(self):
        X, y, _ = make_correlated_data(
            n_samples=30, n_features=100, random_state=0)

        randnorm = stats.norm(loc=0, scale=1)
        q = 0.5
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1]))
        alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

        alphas = alpha_max * alphas_seq / 50

        for fista in [False, True]:
            tol = 1e-10
            w, E, gaps, _ = prox_grad(X, y, alphas, fista=fista)
            with self.subTest():
                self.assertGreater(tol, gaps[-1])


if __name__ == '__main__':
    unittest.main()
