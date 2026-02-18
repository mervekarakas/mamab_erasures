#!/usr/bin/env python
"""
Full K=100 experiments for JSAIT paper — parallelized.
Runs all 4 configs (M=4/20/40 nominal + M=40 hard) concurrently,
each using process-based episode parallelism internally.

Run with: ~/miniconda3/envs/cs260r/bin/python run_k100_full.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import pickle, time
from concurrent.futures import ProcessPoolExecutor

from helper_methods import run_episodes_with_same_erasures
from models import (
    FEEDBACK_ACK_SUCCESS,
    FEEDBACK_NONE,
)

# ── Config ──
K = 100
T_FULL = 500_000
EPISODES_FULL = 100
VAR = 1.0
MU = 'random'
RNG_SEED = 12345
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Algorithms ──
algs = [
    ('SAE', True, 'Scheduled'),                       # SP2 (no feedback)
    ('SAE', True, 'Feedback', FEEDBACK_ACK_SUCCESS),   # SP2-Feedback
    ('SAE', True, 'TPG', FEEDBACK_ACK_SUCCESS),        # TPG
]
fb_list = [a[3] if len(a) == 4 else FEEDBACK_NONE for a in algs]

# ── Epsilon vectors (same sampling logic as experiments_v2) ──
# Must use same RNG sequence as notebook to get identical eps vectors
np.random.seed(RNG_SEED)
Ms = [4, 20, 40]
M_hard = 40

rng_eps = np.random.default_rng(RNG_SEED)

def sample_eps_nominal(M, rng):
    q = M // 4
    n4 = M - 3 * q
    eps1 = rng.uniform(0.1, 0.5, size=q)
    eps2 = rng.uniform(0.5, 0.8, size=q)
    eps3 = rng.uniform(0.8, 0.95, size=q)
    eps4 = rng.uniform(0.95, 0.999, size=n4)
    eps = np.concatenate([eps1, eps2, eps3, eps4])
    eps.sort()
    return eps

def sample_eps_hard_from_nominal(M, eps_nominal, rng):
    q = M // 4
    eps_sorted = np.sort(eps_nominal)
    eps_rest = eps_sorted[q:]
    eps1_new = rng.uniform(0.5, 0.8, size=q)
    eps_hard = np.concatenate([eps1_new, eps_rest])
    eps_hard.sort()
    return eps_hard

eps_nominal = {}
eps_hard_dict = {}
for M in Ms:
    eps_nominal[M] = sample_eps_nominal(M, rng_eps)
eps_hard_dict[M_hard] = sample_eps_hard_from_nominal(M_hard, eps_nominal[M_hard], rng_eps)

# Pre-generate base_actions for each config using the same global RNG sequence
# as the sequential version would have
base_actions_map = {}
for M in Ms:
    base_actions_map[(M, 'nominal')] = np.random.randint(K, size=(M,))
base_actions_map[(M_hard, 'hard')] = np.random.randint(K, size=(M_hard,))


def run_one_config(config):
    """Run a single (M, eps_tag) configuration. Called in a subprocess."""
    M, eps_tag, eps_vec, base_actions = config
    # Enable per-episode parallelism within this config
    # Each config gets ~2 workers for episode parallelism
    os.environ['RUN_MAB_PARALLEL'] = '1'
    os.environ['RUN_MAB_MAX_WORKERS'] = '2'

    label = f"K={K}, M={M}, {eps_tag}"
    print(f"[START] {label}, T={T_FULL}, episodes={EPISODES_FULL}", flush=True)
    t0 = time.time()

    vars_out = run_episodes_with_same_erasures(
        algs,
        iters=T_FULL,
        k=K,
        episodes=EPISODES_FULL,
        m=M,
        var=VAR,
        mu=MU,
        eps=eps_vec,
        base_actions=base_actions,
        feedback_mode=fb_list,
        rng_seed=RNG_SEED,
    )

    elapsed = time.time() - t0
    print(f"[DONE]  {label} in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Compute summaries
    summaries = []
    for alg_key, data in vars_out.items():
        name = data['name']
        ep_regs = data['episode_regrets']
        finals = [np.cumsum(np.sum(ep, axis=1))[-1] for ep in ep_regs]
        mean_r = np.mean(finals)
        std_r = np.std(finals)
        print(f"    {name:>15s}: mean R_T={mean_r:.0f} +/- {std_r:.0f}", flush=True)

        reg = np.sum(data['regret'], axis=1)
        R_T = np.cumsum(reg)[-1]
        TX_T = data['avg_tx'][-1]
        FB_T = data['avg_fb'][-1]
        E_T = 1.0 * TX_T + 0.1 * FB_T
        summaries.append({
            'K': K, 'M': M, 'eps_tag': eps_tag, 'alg': name,
            'R_T': R_T, 'TX_T': TX_T, 'FB_T': FB_T, 'E_T': E_T,
        })

    # Save this config's result individually
    cfg_path = os.path.join(RESULTS_DIR, f'k100_{eps_tag}_M{M}.pkl')
    with open(cfg_path, 'wb') as f:
        pickle.dump({
            'vars_out': vars_out,
            'eps_vec': eps_vec,
            'config': {'K': K, 'T': T_FULL, 'episodes': EPISODES_FULL,
                       'M': M, 'eps_tag': eps_tag},
        }, f)
    print(f"    Saved: {cfg_path}", flush=True)

    return (M, eps_tag), vars_out, summaries


if __name__ == '__main__':
    print(f"K={K}, T={T_FULL}, episodes={EPISODES_FULL}")
    print(f"Ms={Ms}, hard M={M_hard}")
    print(f"Running 4 configs in parallel (each with 2 episode-level workers)")
    print(f"Total CPU cores: {os.cpu_count()}")
    sys.stdout.flush()

    # Build config list
    configs = []
    for M in Ms:
        configs.append((M, 'nominal', eps_nominal[M], base_actions_map[(M, 'nominal')]))
    configs.append((M_hard, 'hard', eps_hard_dict[M_hard], base_actions_map[(M_hard, 'hard')]))

    t_total_start = time.time()

    # Run all 4 configs in parallel (4 top-level processes, each spawns 2 episode workers)
    full_results = {}
    all_summaries = []

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one_config, cfg) for cfg in configs]
        for future in futures:
            key, vars_out, summaries = future.result()
            full_results[key] = vars_out
            all_summaries.extend(summaries)

    # ── Final combined save ──
    save_path = os.path.join(RESULTS_DIR, 'k100_full_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'full_results': full_results,
            'eps_nominal': eps_nominal,
            'eps_hard': eps_hard_dict,
            'config': {
                'T': T_FULL, 'episodes': EPISODES_FULL, 'K': K,
                'VAR': VAR, 'MU': MU, 'RNG_SEED': RNG_SEED,
            }
        }, f)

    summary_df = pd.DataFrame(all_summaries)
    csv_path = os.path.join(RESULTS_DIR, 'k100_full_summary.csv')
    summary_df.to_csv(csv_path, index=False)

    total_elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"Saved: {save_path}")
    print(f"Saved: {csv_path}")
    print(summary_df.to_string(index=False))
