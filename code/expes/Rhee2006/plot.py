from collections import defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt
from slope.plot_utils import configure_plt, _plot_legend_apart

save_fig = True
# save_fig = False

configure_plt()
cmap = plt.get_cmap('tab10')
fig_dir = "../../../figures/"

bench_name = "benchopt_run_2022-08-19_09h03m03.csv"
df = pd.read_csv(bench_name)


solvers = [
    'admm', 'anderson', 'hybrid[cluster_updates=True]',
    'oracle', 'pgd[fista=False]', 'pgd[fista=True]']

dataset_names = ['breheny[dataset=Rhee2006]']
objective_names = df['objective_name'].unique()

obj_col = 'objective_value'


dict_ylabel = {}

dict_legend = {}
dict_legend['admm'] = 'admm'
dict_legend['anderson'] = 'anderson'
dict_legend['hybrid[cluster_updates=True]'] = 'hybrid (ours)'
dict_legend['oracle'] = 'oracle'
dict_legend['pgd[fista=False]'] = 'pgd'
dict_legend['pgd[fista=True]'] = 'fista'

regs = [0.25, 0.1, 0.01]
qs = [0.2, 0.1, 0.05]


dict_xlim = defaultdict(lambda: None, key="default_key")
dict_xlim[0.5] = 0.1
dict_xlim[0.1] = 0.15
dict_xlim[0.01] = 0.8


fontsize = 24
labelsize = 24
regex = re.compile('.*reg=(.*?),')
plt.close('all')
fig, axarr = plt.subplots(
    len(objective_names) // 3,  len(objective_names) // 3,
    sharex=False,
    sharey=True,
    figsize=[12, 6],
    constrained_layout=True)
# handle if there is only 1 objective:
for idx1, reg in enumerate(regs):
    for idx2, q in enumerate(qs):
        objective_name = "SLOPE[fit_intercept=True,q=" + \
            str(q) + ",reg=" + str(reg) + "]"
        df2 = df[df['objective_name'] == objective_name]
        ax = axarr[idx1, idx2]
        # customize here for the floating point
        # c_star = np.min(df2[obj_col]) - 1e-11
        dual0 = (df2[df2['solver_name'] == solvers[0]]
                 ).objective_duality_gap.to_numpy()[0]
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()

            # q1 = df3.groupby('stop_val')['time'].quantile(.1)
            # q9 = df3.groupby('stop_val')['time'].quantile(.9)
            y = curve.objective_duality_gap / dual0
            # y = curve[obj_col] - c_star
            color = cmap(i)
            ax.loglog(
                curve["time"], y, color=color, marker="o", markersize=3,
                label=dict_legend[solver_name], linewidth=3)
        axarr[idx1, idx2].set_xlim(5e-4, dict_xlim[reg])
        axarr[len(regs)-1, idx2].set_xlabel(
            "Time (s)", fontsize=fontsize)

        ax.tick_params(axis='both', which='major', labelsize=labelsize)

        axarr[0, idx2].set_title(r"q=" + str(q), fontsize=fontsize)
    axarr[idx1, 0].set_ylabel(
        r"$\lambda_{\max} / %i  $ " % int(1/reg), fontsize=fontsize)
    axarr[idx1, 0].set_yticks([1, 1e-7])
    axarr[idx1, 0].set_ylim([1e-7, 10])

if save_fig:
    fig.savefig(fig_dir + "Rhee2006.pdf", bbox_inches="tight")
    _plot_legend_apart(axarr[0, 0], fig_dir + "Rhee2006_legend.pdf", ncol=4)
plt.show()
