# Coordinate Descent for SLOPE

This supplement provides code to reproduce the experiments for the academic
research paper _Coordinate Descent for SLOPE_ along with the appendix.

The items in this archive are the following:

- The `code` folder contains the code for the solvers, the results produces from
  our experiments, a few (smaller) experiments, as well as scripts to generate
  the figures in the paper experiments
- The `benchmark` folder contains a [benchopt](https://benchopt.github.io/)
  benchmark for SLOPE, which was used to run the main experiments in the
  paper on simulated and real data.
- The `appendix.pdf` file contains proofs for the paper, additional experiments,
  and other details regarding our work.

## Installation

First make sure that you have
[conda](https://conda.io/projects/conda/en/latest/index.html) available on your
computer. Installation instructions are available
[here](https://conda.io/projects/conda/en/latest/user-guide/install/).

Then, start by creating a conda environment within which the benchmarks should
be run. Here, we also install two R packages that were used in one of the
experiments.

```bash
conda create -n aistats_slopecd -c conda-forge -y \
  python=3.9 r=4.2 r-slope=0.4 r-glmnet=4.1 
conda activate aistats_slopecd
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
benchopt install -y benchmark/
```

## Running the Experiments

Some experiments are available in `code/expes` and can be run simply by calling
`python expes/<experiment>`, or `Rscript expes/<experiment>` where
`<experiment>` is the name of one of the python or R files in the folder.

To re-run the main benchmarks from the paper, modify `benchmark/config.yml` to
include or exclude objectives, solvers, and datasets by commenting or
uncommenting them. Then call

```bash
benchopt run benchmark/ --config benchmark/config.yml
```

to run the benchmark.

## Results

The results used in the paper are stored in the `code/results` folder.

## Figures

The figures can be re-created by calling the python scripts in
`code/scripts/figures`.
