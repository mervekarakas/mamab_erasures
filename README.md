# Multi-Agent Bandit Learning through Heterogeneous Action Erasure Channels

Simulation code for multi-agent multi-armed bandits (MABs) where a central
controller communicates arm selections to distributed agents over independent
erasure channels with heterogeneous erasure probabilities.

## Papers

- **JSAIT 2026**: "Multi-Agent Bandit Learning through Heterogeneous Action Erasure Channels"
  (journal version with feedback modes, Two-Phase Greedy, energy analysis)
- **AISTATS 2024**: "Stochastic Bandits with Heterogeneous Action Erasure Probabilities"
  (conference version with scheduled batch elimination baselines)

## Algorithms

### JSAIT 2026

| Class | Description |
|---|---|
| `BatchSP2` | Offline scheduled batch pulls, no feedback (AISTATS baseline) |
| `BatchSP2Erasure` | SP2 with erasure feedback and stack-based delivery ("SP2-Feedback") |
| `BatchTPG` | Two-Phase Greedy — main JSAIT algorithm |
| `BatchTPGNew` | TPG variant protecting already-delivered arms from takeover |
| `BatchTPGOld` | Earlier TPG with forcible agent unassignment |
| `BatchSP2RRR` | Random round-robin scheduling baseline |
| `BatchSGreedy` | Greedy per-round arm selection baseline |
| `BatchSP2Simplified` | Pointer-based simplified TPG implementation |

### AISTATS 2024 (Legacy)

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

### K=10 experiments (JSAIT main text)
```bash
python experiments/run_k10_mild.py
python experiments/run_k10_bernoulli.py
python experiments/run_k10_movielens.py
```

### K=100 experiments (JSAIT supplement)
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
