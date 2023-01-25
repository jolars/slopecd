#!/usr/bin/python

import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from pyprojroot import here

from slope.data import get_data
from slope.path_utils import setup_path
from slope.solvers import admm, hybrid_cd, newt_alm, prox_grad
from slope.utils import preprocess

dataset = str(sys.argv[1])
solver = str(sys.argv[2])

results_dir = Path(here()) / "results" / "path"
results_dir.mkdir(parents=True, exist_ok=True)
file_path = results_dir / f"{dataset}_{solver}.csv"

if file_path.exists():
    print(f"dataset: {dataset}, solver: {solver} results already exist!")
else:
    X, y = get_data(dataset)

    X = preprocess(X)

    n, p = X.shape
    q = 0.1
    fit_intercept = True

    path_length = 100
    verbose = False
    path_verbose = True
    tol = 1e-6
    max_epochs = 100_000

    intercept = fit_intercept * np.mean(y)
    null_primal = 0.5 * np.linalg.norm(y - intercept) ** 2 / n
    tol_mod = tol * null_primal

    print("Setting up path")
    lambdas, reg = setup_path(
        X,
        y,
        fit_intercept=fit_intercept,
        verbose=path_verbose,
        path_length=path_length,
        tol=tol_mod,
        q=q,
    )

    solver_args = dict(fit_intercept=fit_intercept, max_epochs=max_epochs, tol=tol_mod)

    # Force numba JIT compilation
    print("Running once to JIT compile")
    if solver == "hybrid_cd":
        hybrid_cd(X, y, alphas=lambdas, reg=[0.5], **solver_args)
    elif solver == "fista":
        prox_grad(X, y, alphas=lambdas, fista=True, reg=[0.5], **solver_args)
    elif solver == "anderson":
        prox_grad(X, y, alphas=lambdas, anderson=True, reg=[0.5], **solver_args)
    elif solver == "admm":
        admm(
            X, y, lambdas=lambdas, adaptive_rho=False, rho=100, reg=[0.5], **solver_args
        )
    elif solver == "newt_alm":
        newt_alm(X, y, lambdas=lambdas, reg=[0.5], **solver_args)
    else:
        raise ValueError("there is no solver by that name")

    # time
    print("Starting run")
    t0 = timer()

    if solver == "hybrid_cd":
        hybrid_cd(X, y, alphas=lambdas, **solver_args, reg=reg)
    elif solver == "fista":
        prox_grad(X, y, alphas=lambdas, fista=True, reg=reg, **solver_args)
    elif solver == "anderson":
        prox_grad(X, y, alphas=lambdas, anderson=True, reg=reg, **solver_args)
    elif solver == "admm":
        admm(X, y, lambdas=lambdas, adaptive_rho=False, rho=100, reg=reg, **solver_args)
    elif solver == "newt_alm":
        newt_alm(X, y, lambdas=lambdas, reg=reg, **solver_args)
    else:
        raise ValueError("there is no solver by that name")

    time = timer() - t0

    file_path.write_text(f"{dataset},{solver},{time}\n", encoding="utf-8")

    print(f"dataset: {dataset}, solver: {solver}, time: {time}")
