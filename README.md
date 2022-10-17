# Coordinate Descent for SLOPE

This repository provides code to reproduce the experiments for the academic research paper *Coordinate Descent for SLOPE*.

The code for the project is split into two parts:

- The `code` folder within this directory, which contains the code for the
  solvers, including our contribution
- The `benchmark` folder, which is distributed along with the supplement, which
  contains a [benchopt](https://benchopt.github.io/) benchmark for SLOPE


## Installation

First make sure that you have
[conda](https://conda.io/projects/conda/en/latest/index.html) available on your
computer.

Then, start by creating a conda environment within which the benchmarks should
be run.

```python
conda create -n slope_aistats python=3.10 r=4.2 r-slope=0.4 r-glmnet=4.1
conda activate slope_aistats
pip install benchopt
```

After this, run `pip install code/`

```python
pip install code/
```

To install the benchopt benchmark, run

```python
benchopt install benchmark/
```

## Running the Experiments

Some experiments are available in `code/expes` and can be run simply by calling
`python expes/<experiment>`, or `Rscript expes/<experiment>` where
`<experiment>` is the name of one of the python or R files in the folder. 

To re-run the main benchmarks in the paper, modify `benchmark/config.yml` to
include experiments by uncommenting them, and then call

```python
benchopt run benchmark/ --config benchmark/config.yml
```

## Results

The results used in the paper are stored in the `code/results` folder.

## Figures

The figures can be re-created by calling the python scripts in
`code/scripts/figures`.
