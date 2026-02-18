"""
BatchSP2 — Offline scheduled batch pulls without feedback.

Main no-feedback baseline from AISTATS 2024. Uses successive arm elimination
with a pre-computed pull schedule that accounts for per-channel erasure
probabilities and repetition requirements.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_NONE


class BatchSP2(BanditBase):
    """Offline scheduled batch elimination (SP2) — no erasure feedback.

    Pre-computes a pull schedule accounting for per-channel erasure probabilities
    and repetition factors. Arms are eliminated based on confidence intervals
    after each batch of pulls.
    """
    def __init__(
        self,
        k: int,
        m: int,
        iters: int,
        alphas: Sequence[int],
        var: float = 1,
        c: float = 1,
        mu: Union[str, Sequence[float]] = "random",
        epsilon: Union[float, Sequence[float]] = 0,
        base=None,
        erasure_seq=None,
        feedback_mode: str = FEEDBACK_NONE,
        rng: Optional[Generator] = None,
        verbose: bool = False,
    ):
        super().__init__(
            name="SP2",
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

    def schedule(self, pulls_per_arm: int) -> np.ndarray:
        active_arm_indices = np.where(self.active_arms == 1)[0]
        self.rng.shuffle(active_arm_indices)
        if len(active_arm_indices) == 0:
            return np.empty((0, self.m), dtype=int)

        horizon_adjust = np.sum(pulls_per_arm / (self.alphas - 1 + pulls_per_arm))
        T_i = math.ceil(pulls_per_arm * len(active_arm_indices) / horizon_adjust)
        assigned = np.zeros((self.m, self.k), dtype=int)
        required_pulls = np.zeros(self.k, dtype=int)
        required_pulls[self.active_arms == 1] = pulls_per_arm
        effective_pulls = np.zeros(self.k, dtype=int)

        arm_cursor = 0
        for agent_idx in range(self.m):
            t_end = 0
            p = self.alphas[agent_idx] - 1 + pulls_per_arm
            while t_end + p <= T_i:
                if arm_cursor >= len(active_arm_indices):
                    break
                arm_idx = active_arm_indices[arm_cursor]
                assigned[agent_idx, arm_idx] = p
                effective_pulls[arm_idx] = pulls_per_arm
                t_end += p
                arm_cursor += 1

        remaining_arms = len(active_arm_indices) - arm_cursor
        if remaining_arms > 0:
            chunk_per_arm = max(1, math.floor(self.m / (2 * remaining_arms)))
            chunk_size = math.ceil(pulls_per_arm / chunk_per_arm)
            agents_half = max(1, math.floor(self.m / 2))
            chunk_per_agent = math.ceil(remaining_arms * chunk_per_arm / agents_half)

            agent_idx = 0
            agent_chunk = 0
            for idx in range(arm_cursor, len(active_arm_indices)):
                arm_idx = active_arm_indices[idx]
                while effective_pulls[arm_idx] < required_pulls[arm_idx]:
                    num_chunks = min(chunk_per_agent - agent_chunk, chunk_per_arm)
                    num_pulls = min(pulls_per_arm - effective_pulls[arm_idx], num_chunks * chunk_size)
                    assigned[agent_idx, arm_idx] = self.alphas[agent_idx] - 1 + num_pulls
                    effective_pulls[arm_idx] += num_pulls
                    agent_chunk += num_chunks
                    if agent_chunk >= chunk_per_agent:
                        agent_idx += 1
                        agent_chunk = 0

        max_iter = int(max(np.sum(assigned, axis=1)))
        schedule = -1 * np.ones((max_iter, self.m), dtype=int)
        for agent_idx in range(self.m):
            ctr = 0
            for arm_idx in range(self.k):
                if assigned[agent_idx][arm_idx] > 0:
                    schedule[ctr:ctr + assigned[agent_idx][arm_idx], agent_idx] = arm_idx
                    ctr += assigned[agent_idx][arm_idx]
            if ctr < max_iter:
                schedule[ctr:, agent_idx] = self.rng.choice(np.where(self.active_arms == 1)[0], max_iter - ctr)

        return schedule

    def run(self) -> None:
        counter = 0
        num_pulls = 0
        pulls_per_arm = int(np.ceil(2 * np.log(self.iters * self.m)) / 4)

        for _ in range(self.iters):
            pulls_per_arm *= 4
            active_arm_indices = np.where(self.active_arms == 1)[0]
            reward_sums = np.zeros(self.k)
            pull_counts = np.zeros(self.k, dtype=int)

            schedule_matrix = self.schedule(pulls_per_arm)
            reps = np.zeros(self.m, dtype=int)

            for t in range(schedule_matrix.shape[0]):
                if counter >= self.iters:
                    return

                arms = schedule_matrix[t, :].copy()
                proposed_arms = arms.copy()

                if self.erasure_seq is not None:
                    if np.any(self.erasure_index >= self.erasure_seq.shape[0]):
                        break
                    erasure_bits = self.erasure_seq[self.erasure_index, np.arange(self.m)]
                    self.erasure_index += 1
                else:
                    erasure_bits = (self.rng.random(self.m) < self.eps).astype(int)

                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms, erasure_bits)
                for m_idx in range(self.m):
                    if not tx_mask[m_idx]:
                        if self.pulled_ind[m_idx] >= 0:
                            arms[m_idx] = self.pulled_ind[m_idx]
                        continue
                    if erasure_bits[m_idx]:
                        if self.pulled_ind[m_idx] < 0:
                            arms[m_idx] = self.rng.integers(0, self.k)
                        else:
                            arms[m_idx] = self.pulled_ind[m_idx]
                    else:
                        self.pulled_ind[m_idx] = arms[m_idx]

                if self.bernoulli:
                    reward = self.rng.binomial(1, self.mu[arms]).astype(float)
                else:
                    noise = self.rng.normal(loc=0.0, scale=np.sqrt(self.variance), size=self.m)
                    reward = self.mu[arms] + noise

                self.pulled_reward = reward
                self.pulled_regret = np.max(self.mu) - self.mu[arms]
                self.pulled_ind = arms

                self.regrets[counter, :] = self.pulled_regret
                self.regrets_all[counter] = np.sum(self.pulled_regret)
                self.rewards[counter, :] = reward
                self.rewards_all[counter] = np.sum(reward)

                self.tx_over_time[counter] = self.tx_count_total
                self.fb_over_time[counter] = self.fb_count_total

                for j in range(self.m):
                    a_learner = schedule_matrix[t, j]
                    if reps[j] < self.alphas[j] - 1:
                        reps[j] += 1
                    elif pull_counts[a_learner] < pulls_per_arm:
                        reward_sums[a_learner] += reward[j]
                        pull_counts[a_learner] += 1

                    if t + 1 < schedule_matrix.shape[0] and schedule_matrix[t + 1, j] != schedule_matrix[t, j]:
                        reps[j] = 0

                counter += 1
                if counter >= self.iters:
                    return

            for arm_idx in active_arm_indices:
                self.k_reward_q[arm_idx] = (self.k_reward_q[arm_idx] * num_pulls + reward_sums[arm_idx]) / (num_pulls + pull_counts[arm_idx])
            num_pulls += pulls_per_arm

            for arm_idx in active_arm_indices:
                diff = np.max(self.k_reward_q[active_arm_indices]) - self.k_reward_q[arm_idx]
                if diff > C_CONFIDENCE * self.c * float(np.sqrt(np.log(self.iters * self.m) / (2 * pulls_per_arm))):
                    self.active_arms[arm_idx] = 0

    def reset(self, mu=None, base_actions=None, erasure_seq=None) -> None:
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
