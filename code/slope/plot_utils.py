import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

C_LIST = sns.color_palette("colorblind", 8)
C_LIST_DARK = sns.color_palette("dark", 8)


def configure_plt(fontsize=10, poster=True):
    """Configure matplotlib with TeX and seaborn."""
    rc("font", **{"family": "sans-serif", "sans-serif": ["Computer Modern Roman"]})
    usetex = matplotlib.checkdep_usetex(True)
    params = {
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize - 2,
        "ytick.labelsize": fontsize - 2,
        "text.usetex": usetex,
        "figure.figsize": (8, 6),
    }
    plt.rcParams.update(params)

    sns.set_palette("colorblind")
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")


def _plot_legend_apart(ax, figname, ncol=None, title=None):
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


dict_algo_name = {}


current_palette = sns.color_palette("colorblind")
dict_color = {}
