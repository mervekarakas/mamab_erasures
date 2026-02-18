"""
Vanilla_SAE_ma — Successive Arm Elimination without repetition (legacy AISTATS).

For each batch, calculates the total number of arms and assigns them equally
to all agents. No delay repetitions are used — each pull is counted directly.

Refactored onto BanditBase for consistent RNG seeding and reset API.
"""

from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, FEEDBACK_NONE


class Vanilla_SAE_ma(BanditBase):
    """
    Successive Arm Elimination without repetition for multi-agent setting.
    """
    def __init__(self, k, m, iters, alphas=None, var=1, c=1, mu='random',
                 epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_NONE, rng=None, verbose=False):
        # Legacy class doesn't use alphas; provide dummy ones
        if alphas is None:
            alphas = np.ones(m, dtype=int)
        super().__init__(
            name='MA-SAE',
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

    def pull(self, n_m):
        num_pulls_per_agent = np.ceil(np.sum(self.active_arms) * n_m / self.m)
        assigned = np.zeros((self.m, self.k), dtype=int)
        req_pulls = np.zeros(self.k, dtype=int)
        req_pulls[self.active_arms == 1] = n_m
        ind = 0
        for i in range(self.m):
            c = 0
            while c < num_pulls_per_agent and ind < self.k:
                num_pull = min(num_pulls_per_agent - c, req_pulls[ind])
                req_pulls[ind] -= num_pull
                c += num_pull
                assigned[i, ind] = num_pull
                if req_pulls[ind] == 0:
                    ind += 1

        max_iter = max(np.sum(assigned, axis=1))
        arr_of_schd = -1 * np.ones((max_iter, self.m)).astype(int)

        for i in range(self.m):
            ctr_ = 0
            for j in range(self.k):
                if assigned[i][j] > 0:
                    arr_of_schd[ctr_:ctr_ + assigned[i][j], i] = j
                    ctr_ += assigned[i][j]
            if ctr_ < max_iter:
                arr_of_schd[ctr_:, i] = self.rng.choice(np.where(self.active_arms == 1)[0], max_iter - ctr_)

        return arr_of_schd

    def run(self):
        counter = 0
        num_pulls = 0
        del_i = 2
        for i in range(self.iters):
            del_i /= 2
            active_arm_inds = np.where(self.active_arms == 1)[0]
            n_m = self.c * int(np.ceil(2 * np.log(self.iters * self.m * (del_i ** 2)) / (del_i ** 2)))

            reward_sums = np.zeros(self.k)
            reward_cts = np.zeros(self.k).astype(int)

            arr_of_schd = self.pull(n_m)

            for t in range(arr_of_schd.shape[0]):
                if counter >= self.iters:
                    return

                arms = arr_of_schd[t, :]
                indcs = self.rng.random(self.m) < self.eps
                arms[indcs] = self.pulled_ind[indcs].astype(int)

                noise = np.sqrt(self.variance) * self.rng.normal(0, 1, self.m)
                reward = self.mu[arms] + noise

                self.pulled_reward = reward
                self.pulled_regret = np.max(self.mu) - self.mu[arms]
                self.pulled_ind = arms

                self.regrets[counter, :] = self.pulled_regret
                self.regrets_all[counter] = np.sum(self.pulled_regret)

                self.rewards[counter, :] = reward
                self.rewards_all[counter] = np.sum(reward)

                for j in range(self.m):
                    a = arr_of_schd[t, j]
                    if reward_cts[a] < n_m:
                        reward_sums[a] += reward[j]
                        reward_cts[a] += 1

                counter += 1

            for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a] * num_pulls + reward_sums[a]) / (num_pulls + n_m)
            num_pulls += n_m

            for j in active_arm_inds:
                diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
                if diff > np.sqrt(2 * np.log((del_i ** 2) * self.iters * self.m) / n_m):
                    self.active_arms[j] = 0

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
