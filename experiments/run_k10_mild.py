#!/usr/bin/env python
"""
Quick K=10 run with mild erasure profile (all eps in [0.1, 0.5]).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import pickle, time

os.environ['RUN_MAB_PARALLEL'] = '0'

from helper_methods import run_episodes_with_same_erasures
from models import FEEDBACK_ACK_SUCCESS, FEEDBACK_NONE

K = 10
T = 50_000
EPISODES = 100
VAR = 1.0
MU = 'random'
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


def main():
    all_results = {}

    for M in Ms:
        # Mild profile: all agents sample from Unif[0.1, 0.5]
        eps_vec = np.sort(rng_eps.uniform(0.1, 0.5, size=M))
        base_actions = np.random.randint(K, size=(M,))

        label = f"K={K}, M={M}, mild"
        print(f"[START] {label}, T={T}, episodes={EPISODES}", flush=True)
        t0 = time.time()

        vars_out = run_episodes_with_same_erasures(
            algs,
            iters=T,
            k=K,
            episodes=EPISODES,
            m=M,
            var=VAR,
            mu=MU,
            eps=eps_vec,
            base_actions=base_actions,
            feedback_mode=fb_list,
            rng_seed=RNG_SEED,
        )

        elapsed = time.time() - t0
        print(f"[DONE]  {label} in {elapsed:.0f}s", flush=True)

        for alg_key, data in vars_out.items():
            name = data['name']
            reg = np.sum(data['regret'], axis=1)
            R_T = np.cumsum(reg)[-1]
            print(f"  {name:>15s}: R_T={R_T:.1f}", flush=True)

        all_results[(M, 'mild')] = vars_out

    # Save
    pkl_path = os.path.join(RESULTS_DIR, 'k10_mild_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'full_results': all_results}, f)
    print(f"\nSaved: {pkl_path}", flush=True)


if __name__ == '__main__':
    main()
