#!/usr/bin/env python
"""
Run a single K=100 config with memory-efficient batched episodes.
Runs episodes in small batches, accumulates running mean/std of
cumulative regret curves, and discards per-episode arrays.

Usage:
  python run_k100_single.py <M> <eps_tag>
  e.g.: python run_k100_single.py 40 nominal
        python run_k100_single.py 40 hard
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import pickle, time

os.environ['RUN_MAB_PARALLEL'] = '0'

from helper_methods import run_episodes_with_same_erasures
from models import FEEDBACK_ACK_SUCCESS, FEEDBACK_NONE

# ── Config ──
K = 100
T_FULL = 500_000
EPISODES_FULL = 100
BATCH_SIZE = 5  # run 5 episodes at a time to limit memory
VAR = 1.0
MU = 'random'
RNG_SEED = 12345
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parse args
M = int(sys.argv[1])
eps_tag = sys.argv[2]

# ── Algorithms ──
algs = [
    ('SAE', True, 'Scheduled'),
    ('SAE', True, 'Feedback', FEEDBACK_ACK_SUCCESS),
    ('SAE', True, 'TPG', FEEDBACK_ACK_SUCCESS),
]
fb_list = [a[3] if len(a) == 4 else FEEDBACK_NONE for a in algs]

# ── Epsilon vectors (same RNG sequence as experiments_v2) ──
Ms_all = [4, 20, 40]
M_hard = 40

np.random.seed(RNG_SEED)
rng_eps = np.random.default_rng(RNG_SEED)

def sample_eps_nominal(M_val, rng):
    q = M_val // 4
    n4 = M_val - 3 * q
    eps = np.concatenate([
        rng.uniform(0.1, 0.5, size=q),
        rng.uniform(0.5, 0.8, size=q),
        rng.uniform(0.8, 0.95, size=q),
        rng.uniform(0.95, 0.999, size=n4),
    ])
    eps.sort()
    return eps

def sample_eps_hard_from_nominal(M_val, eps_nom, rng):
    q = M_val // 4
    eps_sorted = np.sort(eps_nom)
    eps_hard = np.concatenate([rng.uniform(0.5, 0.8, size=q), eps_sorted[q:]])
    eps_hard.sort()
    return eps_hard

eps_nominal = {}
for M_val in Ms_all:
    eps_nominal[M_val] = sample_eps_nominal(M_val, rng_eps)
eps_hard_vec = sample_eps_hard_from_nominal(M_hard, eps_nominal[M_hard], rng_eps)

base_actions_map = {}
for M_val in Ms_all:
    base_actions_map[(M_val, 'nominal')] = np.random.randint(K, size=(M_val,))
base_actions_map[(M_hard, 'hard')] = np.random.randint(K, size=(M_hard,))


def main():
    if eps_tag == 'hard':
        eps_vec = eps_hard_vec
    else:
        eps_vec = eps_nominal[M]
    base_actions = base_actions_map[(M, eps_tag)]

    label = f"K={K}, M={M}, {eps_tag}"
    n_batches = EPISODES_FULL // BATCH_SIZE
    print(f"[START] {label}, T={T_FULL}, episodes={EPISODES_FULL}, "
          f"batch_size={BATCH_SIZE}, n_batches={n_batches}", flush=True)

    # We need deterministic per-batch RNG seeds that together reproduce the
    # same results as a single run of EPISODES_FULL episodes.
    # The helper uses rng_seed to generate per-episode seeds, so we generate
    # them all upfront and pass the appropriate slice per batch.
    base_rng = np.random.default_rng(RNG_SEED)
    all_ep_seeds = base_rng.integers(0, 2**32, size=EPISODES_FULL, dtype=np.uint32)

    # Accumulators: Welford's online algorithm for mean and variance
    # Per algorithm: running mean and M2 of cumulative regret curve (shape T,)
    alg_names = {}
    alg_count = {}  # per-algorithm episode count for Welford's
    running_mean = {}   # alg_key -> (T,) array
    running_m2 = {}     # alg_key -> (T,) array (sum of squared deviations)
    # Also accumulate avg regret, tx, fb across all episodes
    total_avg_regret = {}  # alg_key -> (T, M)
    total_avg_tx = {}      # alg_key -> (T,)
    total_avg_fb = {}      # alg_key -> (T,)

    t_total = time.time()

    for batch_idx in range(n_batches):
        t0 = time.time()
        ep_start = batch_idx * BATCH_SIZE
        ep_end = ep_start + BATCH_SIZE

        # Run this batch
        vars_out = run_episodes_with_same_erasures(
            algs,
            iters=T_FULL,
            k=K,
            episodes=BATCH_SIZE,
            m=M,
            var=VAR,
            mu=MU,
            eps=eps_vec,
            base_actions=base_actions,
            feedback_mode=fb_list,
            rng_seed=int(all_ep_seeds[ep_start]),  # seed for this batch
        )

        # Process each algorithm's results
        for alg_key, data in vars_out.items():
            name = data['name']
            alg_names[alg_key] = name

            # Initialize accumulators on first batch
            if alg_key not in running_mean:
                running_mean[alg_key] = np.zeros(T_FULL)
                running_m2[alg_key] = np.zeros(T_FULL)
                total_avg_regret[alg_key] = np.zeros((T_FULL, M))
                total_avg_tx[alg_key] = np.zeros(T_FULL)
                total_avg_fb[alg_key] = np.zeros(T_FULL)
                alg_count[alg_key] = 0

            # Accumulate avg regret/tx/fb (weighted by batch size)
            total_avg_regret[alg_key] += data['regret'] * BATCH_SIZE
            total_avg_tx[alg_key] += data['avg_tx'] * BATCH_SIZE
            total_avg_fb[alg_key] += data['avg_fb'] * BATCH_SIZE

            # Update running mean/std of cumulative regret curves (Welford's)
            for ep_reg in data['episode_regrets']:
                # ep_reg shape (T, M) -> cumulative regret curve shape (T,)
                cum_reg = np.cumsum(np.sum(ep_reg, axis=1))
                alg_count[alg_key] += 1
                n_seen = alg_count[alg_key]
                delta = cum_reg - running_mean[alg_key]
                running_mean[alg_key] += delta / n_seen
                delta2 = cum_reg - running_mean[alg_key]
                running_m2[alg_key] += delta * delta2

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_total
        print(f"  Batch {batch_idx+1}/{n_batches} done in {elapsed:.0f}s "
              f"(total: {total_elapsed/60:.1f}min)", flush=True)

        # Free batch memory
        del vars_out

    # ── Final statistics ──
    n = EPISODES_FULL
    results_out = {}
    summary_rows = []

    for alg_key in alg_names:
        name = alg_names[alg_key]
        mean_curve = running_mean[alg_key]
        std_curve = np.sqrt(running_m2[alg_key] / n)
        avg_regret = total_avg_regret[alg_key] / n
        avg_tx = total_avg_tx[alg_key] / n
        avg_fb = total_avg_fb[alg_key] / n

        final_mean = mean_curve[-1]
        final_std = std_curve[-1]
        print(f"  {name:>15s}: mean R_T={final_mean:.0f} +/- {final_std:.0f}", flush=True)

        results_out[alg_key] = {
            'name': name,
            'regret': avg_regret,
            'mean_cumreg': mean_curve,
            'std_cumreg': std_curve,
            'avg_tx': avg_tx,
            'avg_fb': avg_fb,
        }

        reg = np.sum(avg_regret, axis=1)
        R_T = np.cumsum(reg)[-1]
        TX_T = avg_tx[-1]
        FB_T = avg_fb[-1]
        E_T = 1.0 * TX_T + 0.1 * FB_T
        summary_rows.append({
            'K': K, 'M': M, 'eps_tag': eps_tag, 'alg': name,
            'R_T': R_T, 'TX_T': TX_T, 'FB_T': FB_T, 'E_T': E_T,
        })

    total_elapsed = time.time() - t_total
    print(f"[DONE]  {label} in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)

    # ── Save ──
    cfg_path = os.path.join(RESULTS_DIR, f'k100_{eps_tag}_M{M}.pkl')
    with open(cfg_path, 'wb') as f:
        pickle.dump({
            'results': results_out,
            'eps_vec': eps_vec,
            'config': {'K': K, 'T': T_FULL, 'episodes': EPISODES_FULL,
                       'M': M, 'eps_tag': eps_tag},
        }, f)
    print(f"Saved: {cfg_path}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, f'k100_{eps_tag}_M{M}_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}", flush=True)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
