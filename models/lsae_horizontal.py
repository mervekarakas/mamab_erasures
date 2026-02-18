"""
LSAE_ma_hor — Successive Arm Elimination with horizontal assignment (legacy AISTATS).

Starting from the first arm, the learner assigns it to all agents and moves
onto the next arm when the required number of effective pulls are received.

Refactored onto BanditBase for consistent RNG seeding and reset API.
"""

import random
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_NONE


class LSAE_ma_hor(BanditBase):
    """
    Successive Arm Elimination with repetition and horizontal assignment.
    """
    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random',
                 epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_NONE, rng=None, verbose=False):
        super().__init__(
            name='MA-LSAE-Horizontal',
            k=k,
            m=m,
            iters=iters,
            alphas=alphas,
            var=var,
            c=c,
            mu=mu,
            epsilon=epsilon,
            base=base,
            erasure_seq=erasure_seq,
            feedback_mode=feedback_mode,
            rng=rng,
            verbose=verbose,
        )
        self.k_reward_q = np.zeros(k)

    def run(self):
        counter = 0
        num_pulls = 0
        M_i = 1
        for i in range(self.iters):
            M_i *= 4
            active_arm_inds = np.where(self.active_arms == 1)[0]

            arms_to_be_assigned = list(active_arm_inds)
            assignments = -1 * np.ones(self.m, dtype=int)
            reps = np.zeros(self.m, dtype=int)

            reward_sums = np.zeros(self.k)
            pull_counts = np.zeros(self.k, dtype=int)

            t = 0
            while (counter < self.iters) and (len(arms_to_be_assigned) > 0):
                t += 1
                for j in range(self.m):
                    if assignments[j] == -1:
                        assignments[j] = arms_to_be_assigned[0]
                    elif pull_counts[assignments[j]] >= M_i:
                        if assignments[j] in arms_to_be_assigned:
                            arms_to_be_assigned.remove(assignments[j])
                        reps[j] = 0
                        if len(arms_to_be_assigned) > 0:
                            assignments[j] = arms_to_be_assigned[0]

                    a_learner = assignments[j]
                    if self.rng.random() < self.eps[j]:
                        a = self.pulled_ind[j]
                    else:
                        a = a_learner

                    self.pulled_ind[j] = a
                    self.pulled_regret[j] = np.max(self.mu) - self.mu[a]
                    self.regrets[counter, j] = self.pulled_regret[j]

                    noise = np.sqrt(self.variance) * self.rng.normal(0, 1, 1)
                    reward = (self.mu[a] + noise)[0]
                    if reps[j] < self.alphas[j] - 1:
                        reps[j] += 1
                    else:
                        pull_counts[a_learner] += 1
                        reward_sums[a_learner] += reward

                    self.pulled_reward[j] = reward
                    self.rewards[counter, j] = reward

                self.rewards_all[counter] = np.sum(self.rewards[counter, :])
                self.regrets_all[counter] = np.sum(self.regrets[counter, :])

                counter += 1

            if counter >= self.iters:
                return

            for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a] * num_pulls + reward_sums[a]) / (num_pulls + pull_counts[a])
            num_pulls += M_i

            for j in active_arm_inds:
                diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
                if diff > C_CONFIDENCE * self.c * float(np.sqrt(np.log(self.k * self.iters * self.m) / (2 * M_i))):
                    self.active_arms[j] = 0

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
