import numpy as np


class SortedL1Norm:
    """Sorted L1 Norm
    Define the penalty for SLOPE the Sorted L1 Norm

    Attributes
    ----------
    array
        lam
    """
    def __init__(self, lam):
        """Initalize penalty

        Parameters
        ----------
        lam : array
            Regularization sequence. Should be non-negative and non-increasing.
        """
        if np.any(lam < 0):
            raise ValueError("lambdas should all be positive")

        if np.any(np.diff(lam) > 0):
            raise ValueError("lambdas should all be non-increasing")

        self.lam = lam

    def evaluate(self, beta):
        """Evaluate the penalty

        Parameters
        ----------
        beta : array
            Vector of coefficients.
        """
        return np.sum(self.lam * np.flip(np.sort(np.abs(beta))))

    def prox(self, beta):
        """Compute the sorted L1 proximal operator

        Parameters
        ----------
        beta : array
            the vector of coefficients

        Returns
        -------
        array
            the result of the proximal operator
        """
        beta_sign = np.sign(beta)
        beta = np.abs(beta)
        ord = np.flip(np.argsort(beta))
        beta = beta[ord]

        p = len(beta)

        s = np.empty(p)
        w = np.empty(p)
        idx_i = np.empty(p, np.int64)
        idx_j = np.empty(p, np.int64)

        k = 0

        for i in range(p):
            idx_i[k] = i
            idx_j[k] = i
            s[k] = beta[i] - self.lam[i]
            w[k] = s[k]

            while (k > 0) and (w[k - 1] <= w[k]):
                k = k - 1
                idx_j[k] = i
                s[k] += s[k + 1]
                w[k] = s[k] / (i - idx_i[k] + 1)

            k = k + 1

        for j in range(k):
            d = max(w[j], 0.0)
            for i in range(idx_i[j], idx_j[j] + 1):
                beta[i] = d

        beta[ord] = beta.copy()
        beta = beta * beta_sign

        return beta
