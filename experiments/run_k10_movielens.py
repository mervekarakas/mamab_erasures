#!/usr/bin/env python
"""
K=10 experiment with MovieLens-derived arm means and Bernoulli rewards.

Motivation:
  Reviewer R3 requested experiments on "more challenging or structured reward
  models, or on real-world datasets." We use the MovieLens 100K dataset
  (https://grouplens.org/datasets/movielens/100k/) to derive realistic,
  non-synthetic arm means.

How MovieLens is typically used for MAB:
  Each movie is treated as an arm. The dataset contains 100K user ratings
  (1-5 stars) for 1,682 movies. To get arm means:
    1. Select K movies (usually the most-rated, so empirical means are reliable).
    2. Compute each movie's average rating -> true arm mean.
    3. Simulate bandit pulls using Bernoulli(mean) or Gaussian(mean, var) rewards.
  This is the standard "parametric" approach used in most MAB papers
  (as opposed to the "replay" method which is limited by data size).

What we do:
  1. Take the top 10 most-rated movies from MovieLens 100K.
  2. Normalize mean ratings to [0,1] via (rating - 1) / 4.
  3. Resulting arm means (sorted):
       [0.539, 0.610, 0.610, 0.658, 0.664, 0.701, 0.720, 0.752, 0.789, 0.840]
     These are clustered with small gaps (min gap = 0.0008), making this a
     harder instance than random Unif[0,1] means.
  4. Use Bernoulli(mu_k) rewards (bandit.bernoulli = True).
  5. Overlay our erasure channel with the mild profile (eps ~ Unif[0.1, 0.5]).

Key point:
  The erasure channel is still simulated (no real dataset has erasure channels,
  since this is a novel model). But the arm means come from real user preference
  data, demonstrating that our algorithms work on realistic, structured reward
  distributions — not just synthetic uniform means.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import pickle, time, copy

os.environ['RUN_MAB_PARALLEL'] = '0'

from models import init_bandit, FEEDBACK_ACK_SUCCESS, FEEDBACK_NONE
from utils import generate_erasure_sequence_multi

K = 10
T = 50_000
EPISODES = 100
VAR = 1.0
RNG_SEED = 12345
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# MovieLens 100K: top 10 most-rated movies, mean rating normalized to [0,1]
# Movie IDs: 50, 258, 100, 181, 294, 286, 288, 1, 300, 121
MOVIELENS_MU = np.sort(np.array([
    0.83962264, 0.70088409, 0.78887795, 0.75197239, 0.53917526,
    0.66424116, 0.61035565, 0.71957965, 0.65777262, 0.60955711,
]))

algs = [
    ('SAE', True, 'Scheduled'),
    ('SAE', True, 'Feedback', FEEDBACK_ACK_SUCCESS),
    ('SAE', True, 'TPG', FEEDBACK_ACK_SUCCESS),
]
fb_list = [a[3] if len(a) == 4 else FEEDBACK_NONE for a in algs]

np.random.seed(RNG_SEED)
rng_eps = np.random.default_rng(RNG_SEED)

Ms = [4, 20, 40]


def run_movielens_episodes(algs, fb_list, iters, k, m, episodes, var,
                           mu_val, eps_vec, base_actions, rng_seed):
    """Run episodes with fixed MovieLens means and Bernoulli rewards."""
    avg_regret = {alg: np.zeros((iters, m)) for alg in algs}
    regrets = {alg: [] for alg in algs}
    bandit_names = {}
    avg_tx = {alg: np.zeros(iters) for alg in algs}
    avg_fb = {alg: np.zeros(iters) for alg in algs}

    master_rng = np.random.default_rng(rng_seed)

    for ep in range(episodes):
        sequence = generate_erasure_sequence_multi(iters, m, eps_vec)
        ep_seed = master_rng.integers(0, 2**32)

        for alg_idx, alg in enumerate(algs):
            alg_name, rep, mode = alg[0], alg[1], alg[2]
            fb = fb_list[alg_idx]
            rng_copy = np.random.default_rng(ep_seed)

            bandit = init_bandit(
                alg_name, rep, mode, iters, k, m,
                mu=mu_val.copy(), eps=eps_vec, var=var, delta=0.01, c=1,
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
    print(f"MovieLens arm means (sorted): {MOVIELENS_MU}", flush=True)
    print(f"Optimal arm: mu*={MOVIELENS_MU.max():.4f}, "
          f"min gap={np.diff(MOVIELENS_MU).min():.4f}", flush=True)

    all_results = {}

    for M in Ms:
        eps_vec = np.sort(rng_eps.uniform(0.1, 0.5, size=M))
        base_actions = np.random.randint(K, size=(M,))

        label = f"K={K}, M={M}, movielens"
        print(f"[START] {label}, T={T}, episodes={EPISODES}", flush=True)
        t0 = time.time()

        vars_out = run_movielens_episodes(
            algs, fb_list, iters=T, k=K, m=M, episodes=EPISODES,
            var=VAR, mu_val=MOVIELENS_MU, eps_vec=eps_vec,
            base_actions=base_actions, rng_seed=RNG_SEED,
        )

        elapsed = time.time() - t0
        print(f"[DONE]  {label} in {elapsed:.0f}s", flush=True)

        for alg_key, data in vars_out.items():
            name = data['name']
            reg = np.sum(data['regret'], axis=1)
            R_T = np.cumsum(reg)[-1]
            print(f"  {name:>15s}: R_T={R_T:.1f}", flush=True)

        all_results[(M, 'movielens')] = vars_out

    pkl_path = os.path.join(RESULTS_DIR, 'k10_movielens_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'full_results': all_results, 'mu': MOVIELENS_MU}, f)
    print(f"\nSaved: {pkl_path}", flush=True)


if __name__ == '__main__':
    main()
