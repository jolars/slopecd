import os

import appdirs
import numpy as np
import rpy2.robjects as robjects
from download import download
from libsvmdata import fetch_libsvm
from rpy2.robjects import numpy2ri
from scipy.sparse import csc_array, issparse

breheny_datasets = [
    "bcTCGA",
    "Scheetz2006",
    "Rhee2006",
    "Golub1999",
    "Singh2002",
    "Gode2011",
    "Scholtens2004",
    "pollution",
    "whoari",
    "bcTCGA",
    "Koussounadis2014",
    "Scheetz2006",
    "Ramaswamy2001",
    "Shedden2008",
    "Rhee2006",
    "Yeoh2002",
    "glc-amd",
    "glioma",
    "spam",
]

base_url = "https://s3.amazonaws.com/pbreheny-data-sets/"


def fetch_breheny(dataset: str, min_nnz=0):
    if dataset not in breheny_datasets:
        raise ValueError(
            f"{dataset} is not among available options: {breheny_datasets}"
        )

    base_dir = appdirs.user_cache_dir("slopecd")

    path = os.path.join(base_dir, dataset + ".rds")

    # download raw data unless it is stored in data folder already
    if not os.path.isfile(path):
        url = base_url + dataset + ".rds"
        download(url, path)

    read_rds = robjects.r["readRDS"]
    numpy2ri.activate()

    data = read_rds(path)
    X = data[0]
    y = data[1]

    density = np.sum(X != 0) / X.size

    if density <= 0.2:
        X = csc_array(X)

    if issparse(X) and min_nnz != 0:
        X = X[:, np.diff(X.indptr) >= min_nnz]

    return X, y


def get_data(dataset: str, min_nnz=0):
    if dataset in breheny_datasets:
        return fetch_breheny(dataset, min_nnz=min_nnz)
    else:
        return fetch_libsvm(dataset, min_nnz=min_nnz)
