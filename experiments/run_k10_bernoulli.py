#!/usr/bin/env python
"""
Quick K=10 run with Bernoulli rewards.
Uses the same mild erasure profile as run_k10_mild.py for direct comparison.
Sets bandit.bernoulli=True on each instance before run().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import pickle, time, copy

os.environ['RUN_MAB_PARALLEL'] = '0'

from helper_methods import (
    init_bandit, generate_erasure_sequence_multi, calculate_repetitions,
)
from models import FEEDBACK_ACK_SUCCESS, FEEDBACK_NONE

K = 10
T = 50_000
EPISODES = 100
VAR = 1.0  # UCB confidence parameter (valid: Bernoulli is 1-subgaussian)
RNG_SEED = 12345
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

algs = [
    ('SAE', True, 'Scheduled'),
    ('SAE', True, 'Feedback', FEEDBACK_ACK_SUCCESS),
    ('SAE', True, 'TPG', FEEDBACK_ACK_SUCCESS),
]
fb_list = [a[3] if len(a) == 4 else FEEDBACK_NONE for a in algs]

np.random.seed(RNG_SEED)
rng_eps = np.random.default_rng(RNG_SEED)

Ms = [4, 20, 40]


def run_bernoulli_episodes(algs, fb_list, iters, k, m, episodes, var, eps_vec,
                           base_actions, rng_seed):
    """Run episodes with Bernoulli rewards by setting bandit.bernoulli=True."""
    avg_regret = {alg: np.zeros((iters, m)) for alg in algs}
    regrets = {alg: [] for alg in algs}
    bandit_names = {}
    avg_tx = {alg: np.zeros(iters) for alg in algs}
    avg_fb = {alg: np.zeros(iters) for alg in algs}

    master_rng = np.random.default_rng(rng_seed)

    for ep in range(episodes):
        # Fresh means for each episode
        mu_val = np.sort(np.random.random(k))
        # Ensure means are in [0.01, 0.99] for valid Bernoulli probabilities
        mu_val = np.clip(mu_val, 0.01, 0.99)
        sequence = generate_erasure_sequence_multi(iters, m, eps_vec)
        ep_seed = master_rng.integers(0, 2**32)
        ep_rng = np.random.default_rng(ep_seed)

        for alg_idx, alg in enumerate(algs):
            alg_name, rep, mode = alg[0], alg[1], alg[2]
            fb = fb_list[alg_idx]
            rng_copy = np.random.default_rng(ep_seed)

            bandit = init_bandit(
                alg_name, rep, mode, iters, k, m,
                mu=mu_val, eps=eps_vec, var=var, delta=0.01, c=1,
                base_actions=base_actions, feedback_mode=fb, rng=rng_copy,
            )
            bandit.erasure_seq = sequence.copy()
            bandit.bernoulli = True
            bandit.run()

            bandit_names[alg] = bandit.name
            avg_regret[alg] += bandit.regrets / episodes
            regrets[alg].append(copy.deepcopy(bandit.regrets))
            avg_tx[alg] += bandit.tx_over_time / episodes
            avg_fb[alg] += bandit.fb_over_time / episodes

    return {
        alg: {
            'regret': avg_regret[alg],
            'episode_regrets': regrets[alg],
            'name': bandit_names[alg],
            'avg_tx': avg_tx[alg],
            'avg_fb': avg_fb[alg],
        }
        for alg in algs
    }


def main():
    all_results = {}

    for M in Ms:
        eps_vec = np.sort(rng_eps.uniform(0.1, 0.5, size=M))
        base_actions = np.random.randint(K, size=(M,))

        label = f"K={K}, M={M}, bernoulli"
        print(f"[START] {label}, T={T}, episodes={EPISODES}", flush=True)
        t0 = time.time()

        vars_out = run_bernoulli_episodes(
            algs, fb_list, iters=T, k=K, m=M, episodes=EPISODES,
            var=VAR, eps_vec=eps_vec, base_actions=base_actions,
            rng_seed=RNG_SEED,
        )

        elapsed = time.time() - t0
        print(f"[DONE]  {label} in {elapsed:.0f}s", flush=True)

        for alg_key, data in vars_out.items():
            name = data['name']
            reg = np.sum(data['regret'], axis=1)
            R_T = np.cumsum(reg)[-1]
            print(f"  {name:>15s}: R_T={R_T:.1f}", flush=True)

        all_results[(M, 'bernoulli')] = vars_out

    pkl_path = os.path.join(RESULTS_DIR, 'k10_bernoulli_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'full_results': all_results}, f)
    print(f"\nSaved: {pkl_path}", flush=True)


if __name__ == '__main__':
    main()
