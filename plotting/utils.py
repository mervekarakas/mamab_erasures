"""
plotting.utils — Simple plotting helpers for bandit experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


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
