import numpy as np
from numba import njit


@njit
def compute_grad_hess_sumX(resid, X_data, X_indices, X_indptr, s, cluster, n_samples):
    grad = 0.0
    L = 0.0

    # NOTE(jolars): We treat the length one cluster case separately because it
    # speeds up computations significantly.
    if len(cluster) == 1:
        j = cluster[0]
        start, end = X_indptr[j : j + 2]

        X_sum_vals = X_data[start:end] * s[0]
        X_sum_rows = X_indices[start:end]

        for i, v in enumerate(X_sum_vals):
            grad += v * resid[X_sum_rows[i]]
            L += v * v
    else:
        rows = []
        vals = []

        for k, j in enumerate(cluster):
            start, end = X_indptr[j : j + 2]
            for ind in range(start, end):
                row_ind = X_indices[ind]
                v = s[k] * X_data[ind]
                grad += v * resid[row_ind]

                rows.append(row_ind)
                vals.append(v)

        ord = np.argsort(np.array(rows))
        rows = np.array(rows)[ord]
        vals = np.array(vals)[ord]

        X_sum_rows = []
        X_sum_vals = []

        j = 0
        while j < len(rows):
            start = rows[j]
            end = start

            val = 0.0
            while start == end and j < len(rows):
                val += vals[j]
                j += 1
                end = rows[j]

            L += val * val

            X_sum_rows.append(start)
            X_sum_vals.append(val)

        X_sum_rows = np.array(X_sum_rows)
        X_sum_vals = np.array(X_sum_vals)

    L /= n_samples

    return grad, L, X_sum_vals, X_sum_rows
