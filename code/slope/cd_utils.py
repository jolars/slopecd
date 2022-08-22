import numpy as np
from numba import njit, prange, types
from numba.typed import Dict


@njit(parallel=True)
def sparse_dot_product(a, vals, inds):
    out = 0.0
    for i in prange(len(vals)):
        out += vals[i] * a[inds[i]]
    return out


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

        grad = -sparse_dot_product(resid, X_sum_vals, X_sum_rows)
        L = np.sum(np.square(X_sum_vals))
    else:
        X_sum = Dict.empty(key_type=types.int32, value_type=types.float64)

        for k, j in enumerate(cluster):
            start, end = X_indptr[j : j + 2]
            for ind in range(start, end):
                row_ind = X_indices[ind]
                v = s[k] * X_data[ind]
                grad -= v * resid[row_ind]

                X_sum[row_ind] = X_sum.get(row_ind, 0.0) + v

        # Convert values and keys to arrays
        # TODO(jolars): It is strange that np.array(X_sum.values()) does not
        # work. There should be some better way to do this.
        vals = X_sum.values()
        keys = X_sum.keys()

        X_sum_vals = np.empty(len(vals), dtype=np.double)
        X_sum_rows = np.empty(len(keys), dtype=np.int32)

        for i, val in enumerate(vals):
            X_sum_vals[i] = val

        for i, key in enumerate(keys):
            X_sum_rows[i] = key

        L = np.sum(np.square(X_sum_vals))

    L /= n_samples

    return grad, L, X_sum_vals, X_sum_rows
