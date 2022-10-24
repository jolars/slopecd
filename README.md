# Coordinate Descent for SLOPE

This repository provides code to reproduce the experiments for the academic
research paper _Coordinate Descent for SLOPE_.

This repository contains the following items:

- The `code` folder contains the code for the solvers, the results produces from
  our experiments, a few (smaller) experiments, as well as scripts to generate
  the figures in the paper experiments
- The `tex` folder contains the source code for the paper.
- The `benchmark_slope` folder is a submodule of the benchopt benchmark
  located at <https://github.com/klopfe/benchmark_slope>, which was used
  to run the main benchmarks in the paper.

## Installation

First make sure that you have
[conda](https://conda.io/projects/conda/en/latest/index.html) available on your
computer. Installation instructions are available
[here](https://conda.io/projects/conda/en/latest/user-guide/install/).

Then, start by creating a conda environment within which the benchmarks should
be run. Here, we also install two R packages that were used in one of the
experiments.

```bash
conda create -n slopecd -c conda-forge -y \
  python=3.9 r=4.2 r-slope=0.4 r-glmnet=4.1
conda activate slopecd
pip install benchopt
```

After this, make sure that you have navigated to the root folder of the
extracted archive. Then run

```bash
pip install code/
```

to install the python module `slope`.

Finally, to install the benchopt benchmark, run

```bash
benchopt install -y benchmark_slope/
```

## Running the Experiments

Some experiments are available in `code/expes` and can be run simply by calling
`python expes/<experiment>`, or `Rscript expes/<experiment>` where
`<experiment>` is the name of one of the python or R files in the folder.

To re-run the main benchmarks from the paper, modify `benchmark_slope/config.yml` to
include or exclude objectives, solvers, and datasets by commenting or
uncommenting them. Then call

```bash
benchopt run benchmark_slope/ --config benchmark_slope/config.yml
```

to run the benchmark.

## Results

The results used in the paper are stored in the `code/results` folder.

## Figures

The figures can be re-created by calling the python scripts in
`code/scripts/figures`.
