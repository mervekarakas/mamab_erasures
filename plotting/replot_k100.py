#!/usr/bin/env python
"""Plot K=100 results from batched pickle files (mean_cumreg / std_cumreg)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pickle
import numpy as np
import matplotlib.pyplot as plt

from plotting.utils import (FONT_SIZE, LEGEND_SIZE, DPI, LINEWIDTH,
                             STD_ALPHA, COLORS, LINESTYLES, ALG_ORDER,
                             style_ax)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def load_k100(eps_tag, M):
    path = os.path.join(RESULTS_DIR, f'k100_{eps_tag}_M{M}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_single(eps_tag, M, ax, show_ylabel=True, show_xlabel_label=True,
                show_legend=False, subplot_name=None, x_upper=None, y_upper=None):
    saved = load_k100(eps_tag, M)
    results = saved['results']

    name_to_data = {}
    for alg_key, data in results.items():
        name_to_data[data['name']] = data

    for i, alg_name in enumerate(ALG_ORDER):
        data = name_to_data[alg_name]
        mean = data['mean_cumreg']
        std = data['std_cumreg']
        t_axis = np.arange(1, len(mean) + 1)

        ax.plot(t_axis, mean, linewidth=LINEWIDTH,
                color=COLORS[i], linestyle=LINESTYLES[i], label=alg_name)
        ax.fill_between(t_axis, mean - std, mean + std,
                        color=COLORS[i], alpha=STD_ALPHA)

    if show_ylabel:
        ax.set_ylabel("Regret ($R_t$)", fontsize=FONT_SIZE)

    if subplot_name:
        if show_xlabel_label:
            ax.set_xlabel("Rounds ($t$)\n" + subplot_name, fontsize=FONT_SIZE)
        else:
            ax.set_xlabel("\n" + subplot_name, fontsize=FONT_SIZE)
    elif show_xlabel_label:
        ax.set_xlabel("Rounds ($t$)", fontsize=FONT_SIZE)

    style_ax(ax)
    if x_upper:
        ax.set_xlim([-10, x_upper])
    if y_upper:
        ax.set_ylim([0, y_upper])
    if show_legend:
        ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.15)


# ═══════════════════════════════════════════
# Determine which configs are available
# ═══════════════════════════════════════════
available = []
for eps_tag, M in [('nominal', 4), ('nominal', 20), ('nominal', 40), ('hard', 40)]:
    path = os.path.join(RESULTS_DIR, f'k100_{eps_tag}_M{M}.pkl')
    if os.path.exists(path):
        available.append((eps_tag, M))
        print(f"Found: {path}")

# ═══════════════════════════════════════════
# Plot available nominal configs (3-subplot)
# ═══════════════════════════════════════════
nominal_Ms = [M for tag, M in available if tag == 'nominal']
if nominal_Ms:
    n = len(nominal_Ms)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5.5), dpi=DPI, squeeze=False)
    axes = axes[0]
    subplot_names = [f'({chr(97+i)})' for i in range(n)]

    for idx, M in enumerate(nominal_Ms):
        plot_single('nominal', M, axes[idx],
                    show_ylabel=(idx == 0),
                    show_xlabel_label=(idx == n // 2),
                    show_legend=(idx == n - 1),
                    subplot_name=subplot_names[idx],
                    x_upper=5e5, y_upper=155000)

        if idx != n - 1:
            axes[idx].xaxis.offsetText.set_visible(False)
        if idx != 0:
            axes[idx].yaxis.offsetText.set_visible(False)

    fig.tight_layout()
    # Save with filename matching supplement figure reference
    fig_path = os.path.join(RESULTS_DIR, 'k100_m4-20-40_nominal_jsait.png')
    fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")

# ═══════════════════════════════════════════
# Plot hard if available
# ═══════════════════════════════════════════
if ('hard', 40) in available:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    plot_single('hard', 40, ax, show_legend=True, x_upper=5e5, y_upper=160000)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'k100_m40_hard_jsait.png')
    fig.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")

print("Done")
