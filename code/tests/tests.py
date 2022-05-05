import unittest
from bisect import bisect_right

import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from scipy import stats

import slope
from slope.clusters import get_clusters, update_cluster
from slope.solvers import prox_grad
from slope.utils import dual_norm_slope


class TestPenalty(unittest.TestCase):
    def test_results(self):
        beta = np.array([3, 2.5, 1.2, -4, 0, -0.2])
        lam = np.array([2.7, 2, 1.7, 1.1, 0.8, 0.4])

        pen = slope.SortedL1Norm(lam)

        self.assertAlmostEqual(pen.evaluate(beta), 22.53)

        np.testing.assert_allclose(
            pen.prox(beta), np.array([1.0, 0.8, 0.1, -1.3, 0.0, -0.0])
        )

    def test_assertions(self):
        with self.assertRaises(ValueError):
            slope.SortedL1Norm(np.array([0.1, 0.2]))
        with self.assertRaises(ValueError):
            slope.SortedL1Norm(np.array([0.2, -0.2]))


class TestPGDSolvers(unittest.TestCase):
    def test_convergence(self):
        X, y, _ = make_correlated_data(n_samples=30, n_features=70, random_state=0)

        randnorm = stats.norm(loc=0, scale=1)
        q = 0.5
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, X.shape[1] + 1) * q / (2 * X.shape[1])
        )
        alpha_max = dual_norm_slope(X, y / len(y), alphas_seq)

        alphas = alpha_max * alphas_seq / 50

        for fista in [False, True]:
            tol = 1e-10
            w, E, gaps, _ = prox_grad(
                X, y, alphas, fista=fista, max_epochs=15_000, gap_freq=10, verbose=False
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
            np.testing.assert_array_equal(c_ind, c_ind_true)


if __name__ == "__main__":
    unittest.main()
