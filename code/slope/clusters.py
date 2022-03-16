import numpy as np


class Clusters:
    inds = []
    coefs = []
    starts = []
    ends = []
    sizes = []

    def __init__(self, beta):
        unique, indices, self.sizes = np.unique(
            np.abs(beta), return_inverse=True, return_counts=True
        )
        self.inds = [[] for _ in range(len(unique))]
        for i in range(len(indices)):
            self.inds[indices[i]].append(i)

        self.inds = self.inds[::-1]
        self.coefs = list(unique[::-1])

        self.sizes = list(self.sizes[::-1])
        self.ends = list(np.cumsum(self.sizes))
        self.starts = [self.ends[i] - self.sizes[i] for i in range(len(self.sizes))]

    """Split a Cluster into Two Clusters

    Split the cluster i into two clusters such that the ith cluster becomes
    the cluster given by `left_split` and the (`i + 1`)th cluster the 
    set difference between the current cluster `i` and `left_split`.

    Parameters
    ----------
    i : int
        The cluster to be split
    left_split: list[int]
        The coefficients of cluster `i` to break off into a new cluster
        
    """

    def split(self, i, left_split):
        if set(left_split).isdisjoint(set(self.inds[i])):
            raise ValueError("left_split is not a subset of the cluster")

        right_split = list(set(self.inds[i]) - set(left_split))

        if len(right_split) > 0:
            self.inds = (
                self.inds[0:i] + [left_split] + [right_split] + self.inds[(i + 1) :]
            )
            self.sizes = (
                self.sizes[0:i]
                + [len(left_split)]
                + [len(right_split)]
                + self.sizes[(i + 1) :]
            )
            self.coefs = self.coefs[0 : (i + 1)] + self.coefs[i:]
            self.starts = self.starts[0 : (i + 1)] + self.starts[i:]
            self.ends = self.ends[0 : (i + 1)] + self.ends[i:]
            self.ends[i] = len(left_split)
            self.starts[i + 1] = len(left_split)

    """Merge Two Clusters into One

    Merge cluster `j` into cluster `i`.

    Parameters
    ----------
    i : int
        The cluster to merge into
    j : int
        The cluster to merge
    """

    def merge(self, i, j):
        self.inds[i].extend(self.inds[j])
        self.sizes[i] += self.sizes[j]

        del self.inds[j]
        del self.sizes[j]
        del self.coefs[j]
        del self.starts[j]
        del self.ends[j]

        self.ends = list(np.cumsum(self.sizes))
        self.starts = [self.ends[k] - self.sizes[k] for k in range(len(self.sizes))]

    """Reorder a Cluster

    Move the cluster at position `i` to position `j`.

    Parameters
    ----------
    i : int
        New cluster position
    j : int
        Old cluster position
    coef_new: float
        New coefficient for the cluster
    """

    def reorder(self, i, j, coef_new):
        self.coefs.insert(i, coef_new)
        self.inds.insert(i, self.inds.pop(j))
        self.sizes.insert(i, self.sizes.pop(j))

        del self.coefs[j]

        self.ends = list(np.cumsum(self.sizes))
        self.starts = [self.ends[k] - self.sizes[k] for k in range(len(self.sizes))]

    """Update Coefficient for a Cluster

    Update the coefficient for cluster `i` and reorder the clusters
    accordingly, possibly merging clusters.

    Parameters
    ----------
    i : int
        The cluster to be updated
    new_coef: float
        The new coefficient for the cluster
    """

    def update(self, ind_old, ind_new, coef_new):
        if ind_old == ind_new:
            # same cluster order, only update the coefficient
            self.coefs[ind_new] = coef_new
        elif self.coefs[ind_new] == coef_new:
            # merge with another cluster
            self.merge(ind_new, ind_old)
        else:
            # change position (reorder) cluster
            self.reorder(ind_new, ind_old, coef_new)
