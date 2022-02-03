import unittest

import numpy as np
import slope


class TestPenalty(unittest.TestCase):
    def test_prox(self):
        beta = np.array([3, 2.5, 1.2, -4, 0, -0.2])
        lam = np.array([2.7, 2, 1.7, 1.1, 0.8, 0.4])

        np.sum(np.flip(np.sort(np.abs(beta))) * lam)

        pen = slope.SortedL1Norm(lam)

        self.assertAlmostEqual(pen.evaluate(beta), 22.53)

        self.assertTrue(
            np.allclose(pen.prox(beta),
                        np.array([1.0, 0.8, 0.1, -1.3, 0.0, -0.0]),
                        rtol=1e-12))


if __name__ == '__main__':
    unittest.main()
