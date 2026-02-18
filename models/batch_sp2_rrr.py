"""
BatchSP2RRR — Random round-robin scheduling baseline.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_BEACON


class BatchSP2RRR(BanditBase):
    """
    Demonstration of a chunk-based scheduling approach for multi-agent bandits under erasures.

    Online Phase: (Random round-robin scheduling)
      - Each round, we look at the arms that are pulled less then needed.
      - We propose a random arm among those arms for the next agent.
      - If it is erased, the agent replays its last-received arm.
      - Exactly one pull is credited to whichever arm was actually played.
      - Once an arm hits the needed 4^i successful pulls, it is removed from the available arms.
    """

    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_BEACON, rng: Optional[Generator] = None, verbose: bool = False):
        """
        k: number of arms
        m: number of agents
        iters: total time horizon for demonstration
        mu: can be 'random' or a list/array of length k with the means
        eps: erasure probability
        variance: noise variance in the reward
        chunk_size: how many "pulls" you assign offline for each (agent, arm)
                    in this simple illustration
        """
        # Initializations
        super().__init__(
            name="MA-LSAE-RRR",
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
        self.k_reward_q = np.zeros(k) # Mean reward for each arm



    def _get_erasure_bit(self, m_idx: int) -> int:
        """
        Return the next erasure bit from the predetermined sequence if available;
        otherwise, sample from epsilon.
        """
        if self.erasure_seq is not None:
            bit = self.erasure_seq[self.erasure_index[m_idx], m_idx]
            self.erasure_index[m_idx] += 1
        else:
            bit = int(self.rng.random() < self.eps[m_idx])

        self.tx_count_total += 1
        self.tx_count_per_agent[m_idx] += 1

        self.update_feedback_counts(m_idx, bit)

        return bit


    def run(self):
        """
        Run the chunk-based approach for 'iters' steps or until all arms have
        at least 'needed' effective pulls.
        (In a real "BatchSP2" logic, you'd do 4^i etc. Then partial eliminate, etc.)

        For demonstration, we say: "Stop once each active arm has needed pulls."
        """
        t = 0
        num_pulls = 0
        M_i = int(np.ceil(2 * np.log(self.iters * self.m )) / 4)

        for i in range(self.iters):
            M_i *= 4
            active_arm_inds = np.where(self.active_arms == 1)[0]

            #keep track of effective number of pulls and rewards for the current batch
            reward_sums = np.zeros(self.k)
            pull_counts = np.zeros(self.k, dtype=int)

            arms_with_remaining_pulls = set(active_arm_inds)
            actual_pull_count = {i: 0 for i in arms_with_remaining_pulls}
            while not self._all_arms_done(pull_counts, M_i):
                if t >= self.iters:
                    return

                # We'll store the actual arms played for logging
                reward_vec = np.zeros(self.m)
                regret_vec = np.zeros(self.m)
                noise_vec = np.sqrt(self.variance) * self.rng.normal(size=self.m)

                # Build proposed arms then sample erasures + update comm counters
                proposed_arms = np.zeros(self.m, dtype=int)
                for m in range(self.m):
                    still_need = [a for a in range(self.k)
                                  if self.active_arms[a] == 1
                                  and a in arms_with_remaining_pulls]
                    if len(still_need) == 0:
                        proposed_arms[m] = self.pulled_ind[m] if self.pulled_ind[m] >= 0 else 0
                    else:
                        proposed_arms[m] = self.rng.choice(still_need)

                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms)

                arms_finished_this_round = set()
                for m in range(self.m):
                    proposed_arm = proposed_arms[m]
                    if not tx_mask[m]:
                        # No transmission; keep last delivered if any
                        actual_arm = self.pulled_ind[m] if self.pulled_ind[m] >= 0 else proposed_arm
                    elif erasure_bits[m]:
                        # Transmission erased; stick with previous if available
                        actual_arm = self.pulled_ind[m] if self.pulled_ind[m] >= 0 else proposed_arm
                    else:
                        # Successful delivery
                        actual_arm = proposed_arm
                    self.pulled_ind[m] = actual_arm

                    # 4) Reward
                    actual_pull_count[actual_arm] = actual_pull_count[actual_arm] + 1 if actual_arm in actual_pull_count else 1
                    reward_val = self.mu[actual_arm] + noise_vec[m]
                    reward_vec[m] = reward_val
                    # Regret
                    regret_val = np.max(self.mu) - self.mu[actual_arm]
                    regret_vec[m] = regret_val

                    # 6) If the arm is active and still needs more pulls, increment effective_pulls
                    #    (If you only want to do partial sums up to 'needed', then do so.)
                    if self.active_arms[actual_arm] == 1 and pull_counts[actual_arm] < M_i:
                        pull_counts[actual_arm] += 1
                        reward_sums[actual_arm] += reward_val
                        # Check if we just finished:
                        if pull_counts[actual_arm] >= M_i:
                            arms_finished_this_round.add(actual_arm)

                arms_with_remaining_pulls = arms_with_remaining_pulls.difference(arms_finished_this_round)

                # --- 3) Logging ---
                self.rewards[t, :] = reward_vec
                self.rewards_all[t] = np.sum(reward_vec)
                self.regrets[t, :] = regret_vec
                self.regrets_all[t] = np.sum(regret_vec)

                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total

                t += 1

            for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a]*num_pulls+reward_sums[a])/(num_pulls+pull_counts[a])
            num_pulls += M_i

            for j in active_arm_inds:
                diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
                if diff > C_CONFIDENCE * self.c * float(np.sqrt(np.log(self.iters * self.m) / (2 * M_i))):
                    self.active_arms[j] = 0
            if self.verbose:
                self.logger.debug('end_time=%s pull counts=%s survivors=%s', t, actual_pull_count, np.where(self.active_arms == 1)[0])


    def _all_arms_done(self, pull_counts, needed):
        """
        Check if all active arms have at least 'needed' successful pulls.
        If yes, we can terminate early.
        """
        active_inds = np.where(self.active_arms == 1)[0]
        if len(active_inds) == 0:
            return True
        # if every arm in active_inds has effective_pulls >= needed, done
        for a in active_inds:
            if pull_counts[a] < needed:
                return False
        return True

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        """
        Reset the environment logs, assignment stacks, etc.
        """
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
