"""
plotting.utils — Shared plotting helpers for bandit experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Style constants ──
FONT_SIZE = 24
TICK_SIZE = 22
LEGEND_SIZE = 20
DPI = 300
LINEWIDTH = 3.5
STD_ALPHA = 0.15
COLORS = ["#0072B2", "#e63946", "#2ca02c"]   # blue (SP2), red (SP2-FB), green (TPG)
LINESTYLES = ["solid", ":", "--"]
ALG_ORDER = ["SP2", "SP2-Feedback", "TPG"]
DEFAULT_ALGS = [
    ('SAE', True, 'Scheduled'),
    ('SAE', True, 'Feedback', 'ack_success'),
    ('SAE', True, 'TPG', 'ack_success'),
]


class CleanFormatter(ScalarFormatter):
    """ScalarFormatter that uses %g (no trailing zeros) but keeps the 1e3/1e4 offset."""
    def _set_format(self):
        self.format = '%g'


def style_ax(ax):
    """Apply scientific notation with clean ticks to both axes."""
    for axis in [ax.xaxis, ax.yaxis]:
        fmt = CleanFormatter()
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 1))
        axis.set_major_formatter(fmt)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.offsetText.set_fontsize(TICK_SIZE)
    ax.yaxis.offsetText.set_fontsize(TICK_SIZE)


def get_mean_std(data):
    """Compute mean and std of cumulative regret across episodes."""
    episodes = data['episode_regrets']  # list of (T, M) arrays
    cum_regs = np.array([np.cumsum(np.sum(ep, axis=1)) for ep in episodes])  # (n_eps, T)
    mean = cum_regs.mean(axis=0)
    std = cum_regs.std(axis=0)
    return mean, std


def plot_and_save(results, labels, title='Regret vs Horizon', info=None, colors=None, linestyles=None, log_scale=False, f_name='results.png'):
    clrs = colors if colors is not None else ["black", "red", "green", "purple", "orange", "olive", "darkmagenta", "mediumpurple"]
    lstyles = linestyles if linestyles is not None else ["solid", ":", "--", "-.", (0, (3, 1, 1, 1, 1)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1))]
    mrkrs = ['*', 'o', 'v', '^', '<', '>', 'p', 'x', '|', None]

    plt.figure()
    clr_idx = 0
    for i in range(len(results)):
        if log_scale:
            plt.semilogy(range(1, results[i].shape[0] + 1), np.cumsum(results[i]), linewidth=2, markevery=5000, color=clrs[clr_idx], label=labels[i])
        else:
            plt.plot(range(1, results[i].shape[0] + 1), np.cumsum(results[i]), linewidth=2, markevery=5000, marker=mrkrs[i], color=clrs[clr_idx], label=labels[i])
        clr_idx = (clr_idx + 1) % len(clrs)

    plt.xlabel("Iterations")
    plt.ylabel("Regret R_T")
    title = title if info is None else title + info
    plt.title(title)
    plt.legend()

    plt.savefig(f_name)

    return f_name
