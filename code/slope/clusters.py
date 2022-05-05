import numpy as np
from numba import njit


# numba implementation of np.unique(., return_counts=True) from
# https://github.com/numba/numba/pull/2959
@njit
def unique_counts(x):
    x = np.sort(x.ravel())
    mask = np.empty(x.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = x[1:] != x[:-1]

    unique = x[mask]

    idx = np.nonzero(mask)[0]
    idx = np.append(idx, mask.size)

    counts = np.diff(idx)

    return unique, counts


@njit
def get_clusters(beta):
    p = len(beta)

    c_ptr = np.empty(p + 1, dtype=np.int64)
    c = np.empty(p)

    # c_tmp, counts = np.unique(np.abs(beta), return_counts=True)
    c_tmp, counts = unique_counts(np.abs(beta))
    n_c = len(c_tmp)
    c_ind = np.argsort(np.abs(beta))[::-1]
    counts_cumsum = np.cumsum(counts[::-1])
    c_ptr[: n_c + 1] = np.hstack((np.array([0.0]), counts_cumsum))
    c[:n_c] = c_tmp[::-1]

    # make sure sub-sequences are ordered (not necessary, but maybe helpful
    # for indexing)
    for i in range(len(c_tmp)):
        c_ind[c_ptr[i] : c_ptr[i + 1]].sort()

    return c, c_ptr, c_ind, n_c


@njit
def merge_clusters(c, c_ptr, c_ind, n_c, ind_from, ind_to):
    size_from = c_ptr[ind_from + 1] - c_ptr[ind_from]

    c_ind_from = c_ind[c_ptr[ind_from] : c_ptr[ind_from + 1]].copy()

    if ind_from != ind_to:
        # update c
        c[ind_from : n_c - 1] = c[ind_from + 1 : n_c]

        next_ind = ind_to if ind_to < ind_from else ind_to - 1

        # update c_ind
        if abs(ind_to - ind_from) == 1:
            # with adjacent clusters, we only need to sort the indices
            next_ind = min(ind_to, ind_from)
        elif ind_to < ind_from:
            a = c_ptr[ind_to + 1]
            b = c_ptr[ind_from + 1]

            c_ind[a + size_from : b] = c_ind[a : b - size_from]
            c_ind[a : a + size_from] = c_ind_from
        elif ind_to > ind_from:
            a = c_ptr[ind_from]
            b = c_ptr[ind_to + 1]

            c_ind[a : b - size_from] = c_ind[a + size_from : b]
            c_ind[b - size_from : b] = c_ind_from

        # update c_ptr
        if ind_to < ind_from:
            c_ptr[ind_to + 1 : ind_from] += size_from
            c_ptr[ind_from:n_c] = c_ptr[ind_from + 1 : n_c + 1]
        elif ind_to > ind_from:
            c_ptr[ind_from + 1 : ind_to + 1] -= size_from
            c_ptr[ind_from + 1 : n_c] = c_ptr[ind_from + 2 : n_c + 1]

        c_ind[c_ptr[next_ind] : c_ptr[next_ind + 1]].sort()

        n_c -= 1

    return n_c


@njit
def reorder_cluster(c, c_ptr, c_ind, new_coef, ind_old, ind_new):
    cluster = c_ind[c_ptr[ind_old] : c_ptr[ind_old + 1]].copy()
    w = len(cluster)

    # update c
    if ind_new < ind_old:
        c[ind_new + 1 : ind_old + 1] = c[ind_new:ind_old]
    elif ind_new > ind_old:
        c[ind_old:ind_new] = c[ind_old + 1 : ind_new + 1]

    c[ind_new] = abs(new_coef)

    # update c_ind
    if ind_new < ind_old:
        a = c_ptr[ind_new]
        b = c_ptr[ind_old]
        c_ind[a + w : b + w] = c_ind[a:b]
        c_ind[a : a + w] = cluster
    elif ind_new > ind_old:
        a = c_ptr[ind_old + 1]
        b = c_ptr[ind_new + 1]
        c_ind[a - w : b - w] = c_ind[a:b]
        c_ind[b - w : b] = cluster

    # update c_ptr
    if ind_new < ind_old:
        c_ptr[ind_new + 1 : ind_old + 1] = c_ptr[ind_new:ind_old] + w
        c_ptr[ind_new + 1] = c_ptr[ind_new] + w
    elif ind_new > ind_old:
        c_ptr[ind_old:ind_new] = c_ptr[ind_old + 1 : ind_new + 1] - w
        c_ptr[ind_new] = c_ptr[ind_new + 1] - w


@njit
def update_cluster(c, c_ptr, c_ind, n_c, new_coef, ind_old, ind_new):
    old_coef = c[ind_old]

    if abs(new_coef) != abs(old_coef):
        if abs(new_coef) == c[ind_new]:
            n_c = merge_clusters(c, c_ptr, c_ind, n_c, ind_old, ind_new)
        elif ind_old != ind_new:
            reorder_cluster(c, c_ptr, c_ind, new_coef, ind_old, ind_new)
        else:
            # same position as before, just update the coefficient
            c[ind_old] = abs(new_coef)

    return n_c
