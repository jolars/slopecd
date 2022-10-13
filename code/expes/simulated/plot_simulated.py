import re
from pathlib import Path

from pyprojroot import here
import pandas as pd
import matplotlib.pyplot as plt
from slope.plot_utils import _plot_legend_apart
import numpy as np

from figures import figspec

save_fig = True
# save_fig = False

# configure_plt()
cmap = plt.get_cmap('tab10')

base_dir = here("expes/simulated")

# Simulated setting 1
bench_name = base_dir / "Simulated1.parquet"
df = pd.read_parquet(bench_name, engine='pyarrow')

other_bench_names = [
    "Simulated2.parquet",  # Simulated 2
    "Simulated3.parquet"]  # Simulated 3

for name in other_bench_names:
    df_to_add = pd.read_parquet(base_dir / name, engine='pyarrow')
    df = pd.concat([df, df_to_add], ignore_index=True)

solvers = [
    'oracle',
    'hybrid',
    'admm[adaptive_rho=False,rho=100]',
    'anderson',
    'pgd[fista=False]',
    'pgd[fista=True]',
    'newt_alm',
]

dataset_names = [
    "Simulated[X_density=1.0,density=0.001,n_features=20000,n_samples=200]",
    "Simulated[X_density=1.0,density=0.04,n_features=1000,n_samples=20000]",
    "Simulated[X_density=0.001,density=1e-05,n_features=2000000,n_samples=200]"]

dataset_legends = ["Simulated 1", "Simulated 2", "Simulated 3"]
objective_names = df['objective_name'].unique()

obj_col = 'objective_value'


dict_ylabel = {}

dict_legend = {}
dict_legend['admm[adaptive_rho=False,rho=100]'] = r'ADMM $(\rho=100)$'
dict_legend['anderson'] = 'Anderson PGD'
dict_legend['hybrid'] = 'hybrid (ours)'
dict_legend['oracle'] = 'oracle CD'
dict_legend['pgd[fista=False]'] = 'PGD'
dict_legend['pgd[fista=True]'] = 'FISTA'
dict_legend['newt_alm'] = 'Newt-ALM'


dict_linestyle = {}
dict_linestyle['admm[adaptive_rho=False,rho=100]'] = 'solid'
dict_linestyle['anderson'] = 'solid'
dict_linestyle['hybrid'] = 'solid'
dict_linestyle['oracle'] = 'dashed'
dict_linestyle['pgd[fista=False]'] = 'solid'
dict_linestyle['pgd[fista=True]'] = 'solid'
dict_linestyle['newt_alm'] = 'solid'

dict_col = {}
dict_col['admm[adaptive_rho=False,rho=100]'] = cmap(0)
dict_col['anderson'] = cmap(1)
dict_col['hybrid'] = cmap(2)
dict_col['oracle'] = "black"
dict_col['pgd[fista=False]'] = cmap(3)
dict_col['pgd[fista=True]'] = cmap(4)
dict_col['newt_alm'] = cmap(5)

dict_markers = {}
dict_markers['admm[adaptive_rho=False,rho=100]'] = 'o'
dict_markers['anderson'] = 'o'
dict_markers['hybrid'] = 'o'
dict_markers['oracle'] = ''
dict_markers['pgd[fista=False]'] = 'o'
dict_markers['pgd[fista=True]'] = 'o'
dict_markers['newt_alm'] = 'o'

regs = [0.5, 0.1, 0.02]


dict_xlim = {}
dict_xlim[0, 0.5] = (-0.2, 3)
dict_xlim[0, 0.1] = (-0.2, 10)
dict_xlim[0, 0.02] = (-0.2, 10)
dict_xlim[1, 0.5] = (-0.1, 3)
dict_xlim[1, 0.1] = (-0.1, 3)
dict_xlim[1, 0.02] = (-0.1, 3)
dict_xlim[2, 0.5] = (-1, 50)
dict_xlim[2, 0.1] = (-1, 50)
dict_xlim[2, 0.02] = (-5, 300)

# fontsize = 24
# labelsize = 24
regex = re.compile('.*reg=(.*?),')
plt.close('all')
fig, axarr = plt.subplots(
    len(dataset_names),  len(objective_names),
    sharex=False,
    sharey="row",
    figsize=[figspec.FULL_WIDTH, figspec.FULL_WIDTH * 0.55],
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
            # y = curve.objective_value - c_star
            y = curve[obj_col] - c_star
            color = cmap(i)
            ax.semilogy(
                curve["time"], y, color=dict_col[solver_name], markersize=2,
                label=dict_legend[solver_name],
                marker=dict_markers[solver_name],
                linestyle=dict_linestyle[solver_name])
        axarr[idx1, idx2].set_xlim(dict_xlim[idx1, reg])
        axarr[idx1, 0].set_ylim([1e-8, None])

        # axarr[len(regs)-1, idx2].set_xlabel(
        #     "Time (s)", fontsize=fontsize)

        ax.tick_params(axis='both', which='major')

        axarr[0, idx2].set_title(
            r"$\lambda_{\max} / %i  $ " % int(1/reg))
    axarr[idx1, 2].yaxis.set_label_position("right")
    axarr[idx1, 2].set_ylabel(dataset_legends[idx1], rotation=270, va="bottom")
    # axarr[idx1, 0].set_yticks([1, 1e-4, 1e-8])

fig.supxlabel("Time (s)")
fig.supylabel(r"$P(\beta) - P(\beta^*)$")

if save_fig:
    plt.rcParams["text.usetex"] = True

    figpath = figspec.fig_path("simulated.pdf")
    legendpath = figspec.fig_path("simulated_legend.pdf")

    fig.savefig(figpath, bbox_inches="tight", pad_inches=0.05)
    _plot_legend_apart(axarr[0, 0], legendpath, ncol=4)

# plt.show()
