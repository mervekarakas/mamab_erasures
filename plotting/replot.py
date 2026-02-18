#!/usr/bin/env python
"""Re-generate plots from saved pickle with larger fonts + std bands."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pickle
import numpy as np
import matplotlib.pyplot as plt

from plotting.utils import (FONT_SIZE, TICK_SIZE, LEGEND_SIZE, DPI, LINEWIDTH,
                             STD_ALPHA, COLORS, LINESTYLES, ALG_ORDER,
                             DEFAULT_ALGS, style_ax, get_mean_std)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
with open(os.path.join(RESULTS_DIR, 'full_results.pkl'), 'rb') as f:
    saved = pickle.load(f)

full_results = saved['full_results']
Ms = [4, 20, 40]
algs = DEFAULT_ALGS
x_upper = 1e4


# ═══════════════════════════════════════════
# Plot 1: Nominal (M=4, 20, 40) — 3 subplots
# ═══════════════════════════════════════════
y_upper_nom = 8500
subplot_names = ['(a)', '(b)', '(c)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=DPI)

for subplot_idx, M in enumerate(Ms):
    ax = axes[subplot_idx]
    vars_plot = full_results[(M, 'nominal')]

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

    # x-axis label: "Rounds (t)" only on middle subplot, (a)/(b)/(c) on all
    # Use blank line on (a)/(c) so all subplots have same xlabel height
    if subplot_idx == 1:
        ax.set_xlabel("Rounds ($t$)\n" + subplot_names[subplot_idx], fontsize=FONT_SIZE)
    else:
        ax.set_xlabel("\n" + subplot_names[subplot_idx], fontsize=FONT_SIZE)

    style_ax(ax)
    ax.set_xlim([-10, x_upper])
    ax.set_ylim([0, y_upper_nom])

    # Hide 1e4 offset on all except rightmost
    if subplot_idx != len(Ms) - 1:
        ax.xaxis.offsetText.set_visible(False)
    # Hide 1e3 offset on all except leftmost
    if subplot_idx != 0:
        ax.yaxis.offsetText.set_visible(False)

    if subplot_idx == len(Ms) - 1:
        ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.15)

fig.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'k10_m4-20-40_nominal_jsait.png')
fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
print(f"Saved: {fig_path}")

# ═══════════════════════════════════════════
# Plot 2: Hard (M=40)
# ═══════════════════════════════════════════
y_upper_hard = 8900

vars_plot = full_results[(40, 'hard')]
fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)

name_to_data = {}
for alg in algs:
    data = vars_plot[alg]
    name_to_data[data['name']] = data

for i, alg_name in enumerate(alg_order):
    data = name_to_data[alg_name]
    mean, std = get_mean_std(data)
    t_axis = np.arange(1, mean.shape[0] + 1)

    ax.plot(t_axis, mean, linewidth=linewidth,
            color=clrs[i], linestyle=lstyles[i], label=alg_name)
    ax.fill_between(t_axis, mean - std, mean + std,
                    color=clrs[i], alpha=STD_ALPHA)

ax.set_xlabel("Rounds ($t$)", fontsize=FONT_SIZE)
ax.set_ylabel("Regret ($R_t$)", fontsize=FONT_SIZE)
style_ax(ax)
ax.set_xlim([-10, x_upper])
ax.set_ylim([0, y_upper_hard])
ax.legend(fontsize=LEGEND_SIZE)
ax.grid(linewidth=0.15)

fig.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'k10_m40_eps_hard_jsait.png')
fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
print(f"Saved: {fig_path}")

print("Done — check results/ for updated PNGs")
