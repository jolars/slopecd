from os import mkdir
from pathlib import Path

import numpy as np
from pyprojroot import here
from scipy import sparse

from slope.cv import cv
from slope.data import get_data

fit_intercept = True
path_length = 100
verbosity = 2
n_folds = 5
random_state = 2023
overwrite = False

folder_path = Path(here("results/cv-reg/", warn=False))
if not folder_path.exists():
    folder_path.mkdir(parents=True)

datasets = [
    "Rhee2006",
    "bcTCGA",
    # "Scheetz2006",
    # "rcv1.binary",
    # "news20.binary",
    # "YearPredictionMSD",
]

qs = [0.1, 0.2]

for dataset in datasets:
    for q in qs:
        dataset_path = folder_path / f"{dataset}_q={q}.txt"

        if overwrite or not dataset_path.exists():
            print(f"{dataset}, q: {q}: cross-validating to find best reg value.")
            X, y = get_data(dataset, min_nnz=3)

            res, regs = cv(
                X,
                y,
                fit_intercept=fit_intercept,
                q=q,
                path_length=path_length,
                n_folds=n_folds,
                random_state=random_state,
                verbosity=verbosity,
            )

            which_best = np.argmin(np.mean(res, axis=1))
            best_reg = regs[which_best]

            dataset_path.write_text(str(best_reg))
        else:
            print(f"{dataset}, q: {q}: skipping since results already exist.")
