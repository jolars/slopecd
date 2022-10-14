import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyprojroot import here

from slope import plot_utils

save_fig = True
# save_fig = False

cmap = plt.get_cmap("tab10")

results_dir = here("results/real")

bench_name = results_dir / "Rhee2006.parquet"
df = pd.read_parquet(bench_name, engine="pyarrow")

other_bench_names = [
    "bcTCGA.parquet",  # bcTCGA
    "rcv1.parquet",  # rcv1
    "news20.parquet",  # news20
]

for name in other_bench_names:
    df_to_add = pd.read_parquet(results_dir / name, engine="pyarrow")
    df = pd.concat([df, df_to_add], ignore_index=True)

solvers = [
    "admm[adaptive_rho=False,rho=10]",
    "admm[adaptive_rho=False,rho=100]",
    "admm[adaptive_rho=False,rho=1000]",
    "admm[adaptive_rho=True,rho=10]",
    "hybrid",
]

dataset_names = [
    "breheny[dataset=Rhee2006]",
    "breheny[dataset=bcTCGA]",
    "libsvm[dataset=rcv1.binary]",
    "libsvm[dataset=news20.binary]",
]

dataset_legends = ["Rhee2006", "bcTCGA", "rcv1", "news20"]
objective_names = df["objective_name"].unique()

obj_col = "objective_value"

dict_ylabel = {}

dict_legend = {}
dict_legend["admm[adaptive_rho=False,rho=10]"] = r"ADMM $(\rho=10)$"
dict_legend["admm[adaptive_rho=False,rho=100]"] = r"ADMM $(\rho=100)$"
dict_legend["admm[adaptive_rho=False,rho=1000]"] = r"ADMM $(\rho=1000)$"
dict_legend["admm[adaptive_rho=True,rho=10]"] = "adaptive ADMM"
dict_legend["hybrid"] = "hybrid (ours)"


dict_linestyle = {}
dict_linestyle["admm[adaptive_rho=False,rho=10]"] = "solid"
dict_linestyle["admm[adaptive_rho=False,rho=100]"] = "solid"
dict_linestyle["admm[adaptive_rho=False,rho=1000]"] = "solid"
dict_linestyle["admm[adaptive_rho=True,rho=10]"] = "solid"
dict_linestyle["hybrid"] = "solid"

dict_col = {}
dict_col["admm[adaptive_rho=False,rho=10]"] = cmap(1)
dict_col["admm[adaptive_rho=False,rho=100]"] = cmap(3)
dict_col["admm[adaptive_rho=False,rho=1000]"] = cmap(4)
dict_col["admm[adaptive_rho=True,rho=10]"] = cmap(0)
dict_col["hybrid"] = cmap(2)

dict_markers = {}
dict_markers["admm[adaptive_rho=False,rho=100]"] = "o"
dict_markers["admm[adaptive_rho=False,rho=10]"] = "o"
dict_markers["admm[adaptive_rho=False,rho=1000]"] = "o"
dict_markers["admm[adaptive_rho=True,rho=10]"] = "o"
dict_markers["anderson"] = "o"
dict_markers["hybrid"] = "o"

regs = [0.5, 0.1, 0.02]

dict_xlim = defaultdict(lambda: None, key="default_key")
dict_xlim[0, 0.5] = (-0.001, 0.05)
dict_xlim[0, 0.1] = (-0.005, 0.1)
dict_xlim[0, 0.02] = (-0.01, 0.2)

dict_xlim[1, 0.5] = (-0.5, 10)
dict_xlim[1, 0.1] = (-1, 40)
dict_xlim[1, 0.02] = (-1, 40)

dict_xlim[2, 0.5] = (-0.05, 2)
dict_xlim[2, 0.1] = (-0.1, 6)
dict_xlim[2, 0.02] = (-0.5, 10)

dict_xlim[3, 0.5] = (-1, 50)
dict_xlim[3, 0.1] = (-0.5, 50)
dict_xlim[3, 0.02] = (-1, 350)

regex = re.compile(".*reg=(.*?),")

plt.close("all")

plt.rcParams["text.usetex"] = save_fig

fig, axarr = plt.subplots(
    len(dataset_names),
    len(objective_names),
    sharex=False,
    sharey="row",
    figsize=[plot_utils.FULL_WIDTH, plot_utils.FULL_WIDTH * 0.7],
    constrained_layout=True,
)
# handle if there is only 1 objective:
for idx1, dataset in enumerate(dataset_names):
    df1 = df[df["data_name"] == dataset]
    for idx2, reg in enumerate(regs):
        objective_name = "SLOPE[fit_intercept=True,q=0.1" + ",reg=" + str(reg) + "]"
        df2 = df1[df1["objective_name"] == objective_name]
        ax = axarr[idx1, idx2]

        # customize here for the floating point
        c_star = np.min(df2[obj_col]) - 1e-11
        # dual0 = (df2[df2['solver_name'] == solvers[0]]
        #  ).objective_duality_gap.to_numpy()[0]
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2["solver_name"] == solver_name]
            curve = df3.groupby("stop_val").median()

            # q1 = df3.groupby('stop_val')['time'].quantile(.1)
            # q9 = df3.groupby('stop_val')['time'].quantile(.9)
            # y = curve.objective_duality_gap / dual0
            y = curve[obj_col] - c_star
            color = cmap(i)
            ax.semilogy(
                curve["time"],
                y,
                color=dict_col[solver_name],
                markersize=2,
                label=dict_legend[solver_name],
                marker=dict_markers[solver_name],
                linestyle=dict_linestyle[solver_name],
            )
        axarr[idx1, idx2].set_xlim(dict_xlim[idx1, reg])
        axarr[idx1, idx2].set_ylim([1e-8, None])

        ax.tick_params(axis="both", which="major")

        axarr[0, idx2].set_title(r"$\lambda_{\max} / %i  $ " % int(1 / reg))
    axarr[idx1, 2].yaxis.set_label_position("right")
    axarr[idx1, 2].set_ylabel(dataset_legends[idx1], rotation=270, va="bottom")
    axarr[idx1, 0].set_yticks([1, 1e-4, 1e-8])
fig.supxlabel("Time (s)")
fig.supylabel(r"$P(\beta) - P(\beta^*)$")

if save_fig:
    figpath = plot_utils.fig_path("real_appendix.pdf")
    legendpath = plot_utils.fig_path("real_legend_appendix.pdf")

    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.05)
    plot_utils.plot_legend_apart(axarr[0, 0], legendpath, ncol=3)
else:
    plt.show(block=False)
