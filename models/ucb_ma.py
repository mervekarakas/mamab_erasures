"""
ucb_ma — Multi-Agent UCB (legacy AISTATS algorithm).

UCB algorithm extended to the multi-agent setting without repetition.
Each round, all agents propose the UCB-optimal arm; erasures cause
agents to replay their last successfully received arm.

Refactored onto BanditBase for consistent RNG seeding and reset API.
"""

import copy
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, FEEDBACK_NONE


class ucb_ma(BanditBase):
    """
    UCB algorithm extended to multi-agent setting, no repetition.
    """
    def __init__(self, k, m, iters, alphas=None, var=1, c=1, mu='random',
                 epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_NONE, rng=None, verbose=False):
        # Legacy class doesn't use alphas; provide dummy ones
        if alphas is None:
            alphas = np.ones(m, dtype=int)
        super().__init__(
            name='MA-UCB',
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
        self.n = 1
        self.k_n = np.ones(k).astype(int)
        self.k_reward_q = np.zeros(k)
        self.pulled_once = np.zeros(self.k, dtype=int)

    def pull(self):
        if self.n <= self.k:
            ct = min(self.m, self.k - self.n + 1)
            a_learner = np.arange(self.n - 1, self.n - 1 + ct, dtype=int)
            if a_learner.shape[0] < self.m:
                a_learner = np.concatenate((a_learner, self.rng.integers(self.k, size=(self.m - ct,))))

            is_erasure = self.rng.random(self.m) < self.eps
            arms = copy.deepcopy(a_learner)
            arms[is_erasure] = self.pulled_ind[is_erasure].astype(int)
        else:
            a_learner = np.argmax(self.k_reward_q + self.c * np.sqrt(np.log(1 + self.n * (np.log(self.n) ** 2)) / self.k_n))
            arms = self.pulled_ind.astype(int)
            arms[self.rng.random(self.m) > self.eps] = int(a_learner)
            a_learner = np.full(self.m, a_learner, dtype=int)

        noise = np.sqrt(self.variance) * self.rng.normal(0, 1, self.m)
        reward = self.mu[arms] + noise
        self.pulled_reward = reward
        self.pulled_regret = np.max(self.mu) - self.mu[arms]
        self.n += self.m
        self.pulled_ind = arms

        unique_arms = np.unique(a_learner)
        for a in unique_arms:
            inds = np.where(a_learner == a)[0]
            self.k_reward_q[a] = (self.k_reward_q[a] * (self.k_n[a] - 1) + np.sum(reward[inds])) / (self.k_n[a] - 1 + len(inds))
            self.k_n[a] += len(inds)

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.regrets[i, :] = self.pulled_regret.reshape(1, self.m)
            self.rewards[i, :] = self.pulled_reward.reshape(1, self.m)
            self.regrets_all[i] = np.sum(self.pulled_regret)
            self.rewards_all[i] = np.sum(self.pulled_reward)

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.n = 1
        self.k_n = np.ones(self.k).astype(int)
        self.k_reward_q = np.zeros(self.k)
        self.pulled_once = np.zeros(self.k, dtype=int)
