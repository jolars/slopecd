from collections import defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt
from slope.plot_utils import configure_plt, _plot_legend_apart
import numpy as np

save_fig = True
# save_fig = False

configure_plt()
cmap = plt.get_cmap('tab10')
fig_dir = "../../../figures/"

# Rhee2006
bench_name = ["Rhee2006.parquet"]
df = pd.read_parquet(bench_name, engine='pyarrow')


other_bench_names = [
    "bcTCGA.parquet",  # bcTCGA
    "rcv1.parquet",  # rcv1
    "news20.parquet"]  # news20

for name in other_bench_names:
    df_to_add = pd.read_parquet(name, engine='pyarrow')
    df = pd.concat([df, df_to_add], ignore_index=True)

solvers = [
    'admm', 'anderson', 'hybrid',
    'oracle', 'pgd[fista=False]', 'pgd[fista=True]', 'newt_alm']

dataset_names = [
    "breheny[dataset=Rhee2006]",
    "breheny[dataset=bcTCGA]",
    "libsvm[dataset=rcv1.binary]",
    "libsvm[dataset=news20.binary]"]

dataset_legends = ["Rhee2006", "bcTCGA", "rcv1", "news20"]
objective_names = df['objective_name'].unique()

obj_col = 'objective_value'


dict_ylabel = {}

dict_legend = {}
dict_legend['admm'] = 'admm'
dict_legend['anderson'] = 'anderson pgd'
dict_legend['hybrid'] = 'hybrid (ours)'
dict_legend['oracle'] = 'oracle cd'
dict_legend['pgd[fista=False]'] = 'pgd'
dict_legend['pgd[fista=True]'] = 'fista'
dict_legend['newt_alm'] = 'newt alm'

dict_linestyle = {}
dict_linestyle['admm'] = 'solid'
dict_linestyle['anderson'] = 'solid'
dict_linestyle['hybrid'] = 'solid'
dict_linestyle['oracle'] = 'dashed'
dict_linestyle['pgd[fista=False]'] = 'solid'
dict_linestyle['pgd[fista=True]'] = 'solid'
dict_linestyle['newt_alm'] = 'solid'

regs = [0.5, 0.1, 0.02]


dict_xlim = defaultdict(lambda: None, key="default_key")
dict_xlim[0, 0.5] = (1e-3, 3)
dict_xlim[0, 0.1] = (1e-3, 10)
dict_xlim[0, 0.02] = (1e-3, 10)

dict_xlim[1, 0.5] = (1e-3, 150)
dict_xlim[1, 0.1] = (5e-3, 1000)
dict_xlim[1, 0.02] = (0.01, 1000)

dict_xlim[2, 0.5] = (0.01, 1000)
dict_xlim[2, 0.1] = (0.01, 1000)
dict_xlim[2, 0.02] = (0.01, 1000)

dict_xlim[3, 0.5] = (0.1, 1000)
dict_xlim[3, 0.1] = (0.1, 1000)
dict_xlim[3, 0.02] = (0.1, 1000)

fontsize = 24
labelsize = 24
regex = re.compile('.*reg=(.*?),')
plt.close('all')
fig, axarr = plt.subplots(
    len(dataset_names),  len(objective_names),
    sharex=False,
    sharey=True,
    figsize=[12, 10],
    constrained_layout=True)
# handle if there is only 1 objective:
for idx1, dataset in enumerate(dataset_names):
    df1 = df[df['data_name'] == dataset]
    for idx2, reg in enumerate(regs):
        objective_name = "SLOPE[fit_intercept=True,q=0.1" + \
            ",reg=" + str(reg) + "]"
        df2 = df1[df1['objective_name'] == objective_name]
        ax = axarr[idx1, idx2]

        # customize here for the floating point
        c_star = np.min(df2[obj_col]) - 1e-11
        # dual0 = (df2[df2['solver_name'] == solvers[0]]
        #  ).objective_duality_gap.to_numpy()[0]
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()

            # q1 = df3.groupby('stop_val')['time'].quantile(.1)
            # q9 = df3.groupby('stop_val')['time'].quantile(.9)
            # y = curve.objective_duality_gap / dual0
            y = curve[obj_col] - c_star
            color = cmap(i)
            ax.loglog(
                curve["time"], y, color=color, marker="o", markersize=3,
                label=dict_legend[solver_name], linewidth=3,
                linestyle=dict_linestyle[solver_name])
        axarr[idx1, idx2].set_xlim(dict_xlim[idx1, reg])
        axarr[idx1, idx2].set_ylim([1e-8, 1000])

        # axarr[len(dataset_names)-1, idx2].set_xlabel(
        #     "Time (s)", fontsize=fontsize)

        ax.tick_params(axis='both', which='major', labelsize=labelsize)

        axarr[0, idx2].set_title(
            r"$\lambda_{\max} / %i  $ " % int(1/reg), fontsize=fontsize)
    axarr[idx1, 2].set_ylabel(dataset_legends[idx1], fontsize=fontsize)
    axarr[idx1, 2].yaxis.set_label_position("right")
    # axarr[idx1, 0].set_ylabel("Duality gap", fontsize=fontsize)
    axarr[idx1, 0].set_yticks([1, 1e-4, 1e-8])
    # axarr[idx1, 0].set_ylim([1e-8, 100])
fig.supxlabel("Time (s)", fontsize=fontsize)
fig.supylabel(r"$P(\beta) - P(\beta^*)$", fontsize=fontsize)
if save_fig:
    fig.savefig(fig_dir + "real.pdf", bbox_inches="tight")
    _plot_legend_apart(axarr[0, 0], fig_dir + "real_legend.pdf", ncol=4)
plt.show()
