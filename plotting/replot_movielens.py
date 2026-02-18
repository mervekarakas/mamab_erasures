#!/usr/bin/env python
"""Plot K=10 MovieLens reward results — 3-subplot (M=4, 20, 40)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pickle
import numpy as np
import matplotlib.pyplot as plt

from plotting.utils import (FONT_SIZE, TICK_SIZE, LEGEND_SIZE, DPI, LINEWIDTH,
                             STD_ALPHA, COLORS, LINESTYLES, ALG_ORDER,
                             DEFAULT_ALGS, style_ax, get_mean_std)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
algs = DEFAULT_ALGS
x_upper = 5e4


# Load MovieLens results
with open(os.path.join(RESULTS_DIR, 'k10_movielens_results.pkl'), 'rb') as f:
    saved = pickle.load(f)
full_results = saved['full_results']

Ms = [4, 20, 40]
subplot_names = ['(a)', '(b)', '(c)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=DPI)

for subplot_idx, M in enumerate(Ms):
    ax = axes[subplot_idx]
    vars_plot = full_results[(M, 'movielens')]

    name_to_data = {}
    for alg in algs:
        data = vars_plot[alg]
        name_to_data[data['name']] = data

    for i, alg_name in enumerate(ALG_ORDER):
        data = name_to_data[alg_name]
        mean, std = get_mean_std(data)
        t_axis = np.arange(1, mean.shape[0] + 1)

        ax.plot(t_axis, mean, linewidth=LINEWIDTH,
                color=COLORS[i], linestyle=LINESTYLES[i], label=alg_name)
        ax.fill_between(t_axis, mean - std, mean + std,
                        color=COLORS[i], alpha=STD_ALPHA)

    if subplot_idx == 0:
        ax.set_ylabel("Regret ($R_t$)", fontsize=FONT_SIZE)

    if subplot_idx == 1:
        ax.set_xlabel("Rounds ($t$)\n" + subplot_names[subplot_idx], fontsize=FONT_SIZE)
    else:
        ax.set_xlabel("\n" + subplot_names[subplot_idx], fontsize=FONT_SIZE)

    style_ax(ax)
    ax.set_xlim([-10, x_upper])
    ax.set_ylim([0, 15500])

    if subplot_idx != len(Ms) - 1:
        ax.xaxis.offsetText.set_visible(False)
    if subplot_idx != 0:
        ax.yaxis.offsetText.set_visible(False)

    if subplot_idx == len(Ms) - 1:
        ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.15)

fig.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'k10_movielens_jsait.png')
fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
print(f"Saved: {fig_path}")
print("Done")
