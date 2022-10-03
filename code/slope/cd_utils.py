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
def compute_grad_hess_sumX(
    resid,
    X_data,
    X_indices,
    X_indptr,
    X_squared_col_sums,
    previously_active,
    s,
    cluster,
    n_samples
):
    grad = 0.0
    L = 0.0

    # NOTE(jolars): We treat the length one cluster case separately because it
    # speeds up computations significantly.
    if len(cluster) == 1:
        j = cluster[0]
        start, end = X_indptr[j : j + 2]

        X_sum_vals = X_data[start:end] * s[0]
        X_sum_inds = X_indices[start:end]

        if not previously_active[j]:
            X_squared_col_sums[j] = np.sum(np.square(X_sum_vals))
            previously_active[j] = True

        L = X_squared_col_sums[j]

        grad = -sparse_dot_product(resid, X_sum_vals, X_sum_inds)
    else:
        # NOTE(jolars): It is possible to do this even more efficiently by just
        # using arrays and only advancing positions for the array with the
        # lowest index.
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
        inds = X_sum.keys()

        X_sum_vals = np.empty(len(vals), dtype=np.double)
        X_sum_inds = np.empty(len(inds), dtype=np.int32)

        for i, val in enumerate(vals):
            X_sum_vals[i] = val

        for i, ind in enumerate(inds):
            X_sum_inds[i] = ind

        L = np.sum(np.square(X_sum_vals))

    L /= n_samples

    return grad, L, X_sum_vals, X_sum_inds
