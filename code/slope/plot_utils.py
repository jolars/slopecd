import os
from pathlib import Path

import matplotlib.pyplot as plt
from pyprojroot import here

FULL_WIDTH = 6.75
HALF_WIDTH = 3.25


def fig_path(x):
    code_dir = Path(here())
    root_dir = code_dir.parent
    fig_dir = root_dir / "tex" / "figures"

    return fig_dir / x


def table_path(x):
    code_dir = Path(here())
    root_dir = code_dir.parent
    fig_dir = root_dir / "tex" / "tables"

    return fig_dir / x


def plot_legend_apart(ax, figname, ncol=None, title=None):
    """Plot legend apart from figure."""
    # Do all your plots with fig, ax = plt.subplots(),
    # don't call plt.legend() at the end but this instead
    plt.rcParams["text.usetex"] = True

    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 30), constrained_layout=True)
    fig.legend(
        ax.lines,
        [line.get_label() for line in ax.lines],
        ncol=ncol,
        title=title,
        loc="upper center",
    )
    fig.savefig(figname, bbox_inches="tight")
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig
