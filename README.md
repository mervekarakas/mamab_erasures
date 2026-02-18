# Fundamental Limits of Learning Under Erasure-Constrained Communication Channels

Simulation code for multi-agent multi-armed bandits (MABs) where a central
controller communicates arm selections to distributed agents over independent
erasure channels with heterogeneous erasure probabilities.

## Papers

This repository accompanies the following papers:

- **[1]** Journal paper (under submission) — extends [2] with erasure feedback
  modes, Two-Phase Greedy scheduling, and communication energy analysis.
- **[2]** Multi-agent MAB over heterogeneous erasure channels (AISTATS 2024).
- **[3]** Single-agent MAB with erasure feedback (ISIT 2025).
- **[4]** Single-agent MAB over erasure channels — prior work (ISIT 2023).

## Algorithms

### Journal [1]

| Class | Description |
|---|---|
| `BatchSP2` | Offline scheduled batch pulls, no feedback (baseline from [2]) |
| `BatchSP2Erasure` | SP2 with erasure feedback and stack-based delivery ("SP2-Feedback") |
| `BatchTPG` | Two-Phase Greedy — main algorithm |
| `BatchTPGNew` | TPG variant protecting already-delivered arms from takeover |
| `BatchTPGOld` | Earlier TPG with forcible agent unassignment |
| `BatchSP2RRR` | Random round-robin scheduling baseline |
| `BatchSGreedy` | Greedy per-round arm selection baseline |
| `BatchSP2Simplified` | Pointer-based simplified TPG implementation |

### AISTATS [2] (Legacy)

| Class | Description |
|---|---|
| `ucb_ma` | Multi-Agent UCB |
| `LSAE_ma_hor` | Successive Arm Elimination with horizontal assignment |
| `LSAE_ma_ver` | Successive Arm Elimination with vertical assignment |
| `Vanilla_SAE_ma` | Successive Arm Elimination without repetition |

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from models import BatchTPG, BatchSP2, calculate_repetitions, FEEDBACK_ACK_SUCCESS
from runners import run_episodes_with_same_erasures
import numpy as np

K, M, T = 10, 4, 50000
eps = np.array([0.3, 0.6, 0.8, 0.95])
alphas = calculate_repetitions(eps, T, m=M)

algs = [
    ('SAE', True, 'Scheduled'),
    ('SAE', True, 'Feedback', FEEDBACK_ACK_SUCCESS),
    ('SAE', True, 'TPG', FEEDBACK_ACK_SUCCESS),
]

results = run_episodes_with_same_erasures(
    algs, iters=T, k=K, episodes=10, m=M, var=1, mu='random',
    eps=eps, base_actions=np.zeros(M, dtype=int),
    feedback_mode=['none', 'ack_success', 'ack_success'],
    rng_seed=42,
)
```

## Reproducing Paper Results

### K=10 experiments (journal main text)
```bash
python experiments/run_k10_mild.py
python experiments/run_k10_bernoulli.py
python experiments/run_k10_movielens.py
```

### K=100 experiments (journal supplement)
```bash
python experiments/run_k100_single.py 4 nominal
python experiments/run_k100_single.py 20 nominal
python experiments/run_k100_single.py 40 nominal
python experiments/run_k100_single.py 40 hard
```

### Plotting
```bash
python plotting/replot.py           # K=10 nominal + hard
python plotting/replot_mild.py      # K=10 mild erasure
python plotting/replot_bernoulli.py # K=10 Bernoulli rewards
python plotting/replot_movielens.py # K=10 MovieLens rewards
python plotting/replot_k100.py      # K=100 supplement figures
```

## Directory Structure

```
models/           Algorithm implementations (one class per file) + init_bandit factory
experiments/      Experiment runner scripts
plotting/         Replot scripts for generating paper figures + utils.py
notebooks/        Jupyter notebooks (smoke test, legacy, paper experiments)
utils.py          Erasure sequence generation functions
runners.py        Episode runners (run_episodes, run_episodes_with_same_erasures)
helper_methods.py Backward-compat shim (re-exports from utils, runners, models, plotting)
results/          Output directory for .pkl results and figures (gitignored)
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

## References

**[1]** M. Karakas, O. A. Hanna, L. F. Yang, and C. Fragouli, "Fundamental Limits of
Learning Under Erasure-Constrained Communication Channels," under submission.

**[2]** O. A. Hanna\*, M. Karakas\*, L. Yang, and C. Fragouli, "Multi-Agent Bandit Learning
through Heterogeneous Action Erasure Channels," in *Proc. AISTATS*, 2024.
[proceedings](https://proceedings.mlr.press/v238/hanna24a.html)

**[3]** M. Karakas, O. Hanna, L. F. Yang, and C. Fragouli, "Does Feedback Help in
Bandits with Arm Erasures?," in *Proc. IEEE ISIT*, Ann Arbor, MI, 2025.
[IEEE](https://ieeexplore.ieee.org/document/11195223)

**[4]** O. A. Hanna\*, M. Karakas\*, L. F. Yang, and C. Fragouli, "Multi-Arm Bandits over
Action Erasure Channels," in *Proc. IEEE ISIT*, Taipei, Taiwan, 2023.
[IEEE](https://ieeexplore.ieee.org/abstract/document/10206591)

\* Equal contribution.
