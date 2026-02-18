"""
BatchSGreedy — Greedy per-round arm selection baseline.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_BEACON


class BatchSGreedy(BanditBase):
    """
    A 'greedy' batch-based scheduling approach for multi-agent bandits under erasures.

    In each batch i:
      - We require each active arm to be pulled M_i = 4^i times (for demonstration),
        or until partial elimination occurs.
      - In each round of the batch:
          1) Identify all arms that haven't reached M_i pulls in this batch.
          2) If # such arms > m, pick the m arms with the fewest pulls so far (tie-break random).
             If # such arms < m, we replicate these arms to fill the agents (some arms assigned to multiple agents).
          3) Assign one chosen arm (or replicate) to each agent (random assignment).
          4) Each agent attempts to 'receive' its assigned arm:
             - If erased, the agent replays its last successfully received arm.
             - We credit exactly one pull to whichever arm is actually played.
          5) Once an arm meets M_i pulls, remove it from the set of "unfinished arms."
      - After the batch completes, update global empirical means and do an elimination step.
    """

    def __init__(self, k, m, iters, alphas, var=1, c=1,
                 mu='random', epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_BEACON, rng: Optional[Generator] = None, verbose: bool = False):
        super().__init__(
            name='BatchSGreedy',
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

        # Empirical means and counters
        self.k_reward_q = np.zeros(k)      # Empirical means
        self.arm_counts = np.zeros(k, int) # total pulls so far
        self.arm_sums   = np.zeros(k)      # total reward sums so far



    def _get_erasure_bit(self, agent_idx: int) -> int:
        """
        Return the next erasure bit for agent_idx from the predetermined sequence if available;
        otherwise, fall back on a random draw from epsilon.
        """
        if self.erasure_seq is not None:
            bit = self.erasure_seq[self.erasure_index[agent_idx], agent_idx]
            self.erasure_index[agent_idx] += 1
        else:
            bit = int(self.rng.random() < self.eps[agent_idx])

        self.tx_count_total += 1
        self.tx_count_per_agent[agent_idx] += 1

        self.update_feedback_counts(agent_idx, bit)

        return bit


    def run(self):
        """
        Main execution loop. Proceeds in batches, each requiring 4^i effective pulls for active arms.
        In each round, we select arms greedily (those that are not done, picking the fewest-pulled if #arms > m),
        assign them to agents (replicating if #arms < m), handle erasures, and credit pulls.
        We stop either when all arms are eliminated or we hit self.iters total rounds.
        """
        t = 0  # global time/round index

        # Start with initial batch size: M_i ~ 2 * ln(T*m)/4, then multiply by 4 each iteration
        M_i = int(np.ceil(2 * np.log(self.iters * self.m)) / 4)

        # Loop over batches until we run out of time or arms
        while t < self.iters and np.any(self.active_arms == 1):
            M_i *= 4
            # Identify currently active arms
            active_arm_inds = np.where(self.active_arms == 1)[0]
            if len(active_arm_inds) == 0:
                break

            # Local counters for this batch only
            actual_pull_count = {arm: 0 for arm in active_arm_inds}
            pull_counts_batch = {arm: 0 for arm in active_arm_inds}  # how many pulls each arm has in this batch
            pull_sums_batch   = {arm: 0.0 for arm in active_arm_inds}

            # As long as there are arms that haven't reached M_i pulls (and we have time):
            def unfinished_arms():
                return [a for a in active_arm_inds if pull_counts_batch[a] < M_i]

            while len(unfinished_arms()) > 0 and t < self.iters:
                # 1) Collect arms that haven't yet reached M_i
                arms_needed = unfinished_arms()
                K_arms = len(arms_needed)

                # 2) If K_arms > m, pick the m arms with the fewest pulls in this batch
                if K_arms > self.m:
                    # Shuffle to break ties randomly, then select m smallest via argpartition
                    perm = self.rng.permutation(K_arms)
                    arms_perm = np.array(arms_needed)[perm]
                    counts_perm = np.array([pull_counts_batch[a] for a in arms_perm])
                    idx = np.argpartition(counts_perm, self.m)[:self.m]
                    chosen_arms = arms_perm[idx].tolist()
                # If K_arms <= m, replicate them across m agents
                else:
                    chosen_arms = arms_needed  # we'll replicate below if needed

                # 3) Assign arms to agents
                #    We'll build an array assigned_arm[m], where assigned_arm[i] = chosen_arm for agent i
                assigned_arm = np.zeros(self.m, dtype=int)
                if len(chosen_arms) == self.m:
                    # Perfectly matched: 1 arm per agent
                    # Let's do a random permutation to distribute them
                    perm = list(range(self.m))
                    self.rng.shuffle(perm)
                    for i, arm in zip(perm, chosen_arms):
                        assigned_arm[i] = arm
                elif len(chosen_arms) < self.m:
                    # We replicate arms to fill m agents
                    # If K_arms = len(chosen_arms) > 0, each arm is assigned to roughly m/K_arms agents
                    # We'll just do a simple approach: assign them in a round-robin or random manner
                    if len(chosen_arms) == 0:
                        # Edge case: no arms needed => skip
                        break
                    # We'll shuffle the agents, then assign arms in a repeating pattern
                    perm = list(range(self.m))
                    self.rng.shuffle(perm)
                    # pattern index
                    idx = 0
                    for agent_idx in perm:
                        assigned_arm[agent_idx] = chosen_arms[idx % len(chosen_arms)]
                        idx += 1
                else:
                    # len(chosen_arms) > m handled above
                    pass  # already assigned in that block

                # 4) Transmission Phase
                proposed_arms = np.array(assigned_arm, dtype=int)
                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms)
                actual_arms = np.empty(self.m, dtype=int)

                for agent_idx in range(self.m):
                    proposed_arm = proposed_arms[agent_idx]

                    if not tx_mask[agent_idx]:
                        actual_arm = self.pulled_ind[agent_idx] if self.pulled_ind[agent_idx] >= 0 else proposed_arm
                    elif erasure_bits[agent_idx]:
                        actual_arm = self.pulled_ind[agent_idx] if self.pulled_ind[agent_idx] >= 0 else proposed_arm
                    else:
                        actual_arm = proposed_arm
                        self.pulled_ind[agent_idx] = proposed_arm  # update last successfully received arm
                    actual_arms[agent_idx] = actual_arm

                # 5) Reward & Pull Counting (vectorized noise)
                noise_vec = np.sqrt(self.variance) * self.rng.normal(size=self.m)
                reward_vec = self.mu[actual_arms] + noise_vec
                regret_vec = np.max(self.mu) - self.mu[actual_arms]

                for agent_idx in range(self.m):
                    actual_arm = actual_arms[agent_idx]
                    r_val = reward_vec[agent_idx]
                    actual_pull_count[actual_arm] = actual_pull_count[actual_arm] + 1 if actual_arm in actual_pull_count else 1
                    # If the arm is indeed in the batch scope and not done:
                    if actual_arm in pull_counts_batch and pull_counts_batch[actual_arm] < M_i:
                        pull_counts_batch[actual_arm] += 1
                        pull_sums_batch[actual_arm]   += r_val

                # 6) Logging for this round
                self.rewards[t, :] = reward_vec
                self.regrets[t, :] = regret_vec
                self.rewards_all[t] = np.sum(reward_vec)
                self.regrets_all[t] = np.sum(regret_vec)
                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total

                t += 1
                if t >= self.iters:
                    break

            # 7) End of the batch: fold local sums/cnts into global sums
            for arm in active_arm_inds:
                cnt = pull_counts_batch[arm]
                if cnt > 0:
                    self.arm_sums[arm]   += pull_sums_batch[arm]
                    self.arm_counts[arm] += cnt
                    self.k_reward_q[arm]  = self.arm_sums[arm] / float(self.arm_counts[arm])

            # 8) Elimination Step
            #    If an arm is far enough below the best observed mean, we eliminate it.
            survivors = np.where(self.active_arms == 1)[0]
            if len(survivors) == 0:
                break
            best_mean = np.max(self.k_reward_q[survivors])
            # Confidence threshold
            thresh = C_CONFIDENCE * self.c * math.sqrt(np.log(self.iters*self.m) / (2.0*M_i))
            for arm in survivors:
                diff = best_mean - self.k_reward_q[arm]
                if diff > thresh:
                    self.active_arms[arm] = 0
            if self.verbose:
                self.logger.debug('end_time=%s pull counts=%s survivors=%s', t, actual_pull_count, np.where(self.active_arms == 1)[0])
            # End of one batch

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        """
        Reset environment logs while keeping fundamental settings.
        """
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)

        self.k_reward_q = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k, dtype=int)
        self.arm_sums   = np.zeros(self.k)


    def _all_arms_done(self, pull_counts, needed):
        """
        Check if all active arms have at least 'needed' successful pulls.
        If yes, we can terminate early.
        (If you want partial usage of this method, just replicate from other classes.)
        """
        active_inds = np.where(self.active_arms == 1)[0]
        if len(active_inds) == 0:
            return True
        for a in active_inds:
            if pull_counts[a] < needed:
                return False
        return True
