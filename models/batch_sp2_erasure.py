"""
BatchSP2Erasure — SP2 with erasure feedback and stack-based delivery
('SP2-Feedback' in JSAIT paper).
"""

import math
from collections import deque
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_BEACON


class BatchSP2Erasure(BanditBase):
    """SP2 with erasure feedback and stack-based delivery (SP2-Feedback).

    Offline: assigns (agent, arm) pull counts to stacks.
    Online: proposes top-of-stack arm each round; on erasure, replays last
    successful arm. Feedback lets the controller track delivery and prune stacks.
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
            name="SP2-Feedback",
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
        self.assignments = [[] for _ in range(self.m)] #assigned pulls for agents

    def _offline_chunk_assignments(self, M_i):
        """
        Initial offline scheduling to assign number of pulls of active arms to agents
        """
        active_arm_inds = np.where(self.active_arms == 1)[0]
        #random.shuffle(active_arm_inds)

        T_i = math.ceil(M_i * len(active_arm_inds) / np.sum(M_i / (self.alphas - 1 + M_i)))
        max_slots = max(1, len(active_arm_inds))
        arm_ids = -1 * np.ones((self.m, max_slots), dtype=int)
        remaining = np.zeros((self.m, max_slots), dtype=int)
        stack_len = np.zeros(self.m, dtype=int)

        ind = 0  # start with the first active arm
        for i in range(self.m):
            t_end = 0
            p = self.alphas[i] - 1 + M_i
            while t_end + p <= T_i and ind < len(active_arm_inds):
                arm_idx = active_arm_inds[ind]
                arm_ids[i, stack_len[i]] = arm_idx
                remaining[i, stack_len[i]] = p
                stack_len[i] += 1
                t_end += p
                ind += 1

        k_hat = len(active_arm_inds) - ind  # remaining unassigned arms
        if k_hat > 0:
            chunk_per_arm = max(1, math.floor(self.m / (2 * k_hat)))
            chunk_size = math.ceil(M_i / chunk_per_arm)
            agents_half = max(1, math.floor(self.m / 2))
            chunk_per_agent = math.ceil(k_hat * chunk_per_arm / agents_half)

            agent_idx = 0
            agent_chk = 0

            for i in range(ind, len(active_arm_inds)):
                arm_idx = active_arm_inds[i]
                eff_pulls = 0
                req_pulls = M_i
                while eff_pulls < req_pulls and agent_idx < self.m:
                    num_chks = min(chunk_per_agent - agent_chk, chunk_per_arm)
                    num_pulls = min(M_i - eff_pulls, num_chks * chunk_size)
                    arm_ids[agent_idx, stack_len[agent_idx]] = arm_idx
                    remaining[agent_idx, stack_len[agent_idx]] = self.alphas[agent_idx] - 1 + num_pulls
                    stack_len[agent_idx] += 1
                    eff_pulls += num_pulls
                    agent_chk += num_chks

                    if agent_chk >= chunk_per_agent:
                        agent_idx += 1
                        agent_chk = 0

        schedule = []
        for i in range(self.m):
            stack = []
            for j in range(stack_len[i]):
                stack.append((int(arm_ids[i, j]), int(remaining[i, j])))
            schedule.append(deque(stack[::-1]))

        return schedule

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
            actual_pull_counts = {a: 0 for a in active_arm_inds}
            self.assignments = self._offline_chunk_assignments(M_i)

            batch_agent_arm_counts = np.zeros((self.k, self.m), dtype=int)
            # Reuse per-round buffers to cut allocations
            reward_vec = np.zeros(self.m)
            regret_vec = np.zeros(self.m)
            erasure_bits = None  # defer allocation to helper
            noise_vec = np.empty(self.m)
            proposed_arms = np.empty(self.m, dtype=int)

            while not self._all_arms_done(pull_counts, M_i):
                if t >= self.iters:
                    return

                # We'll store the actual arms played for logging
                if not self.bernoulli:
                    noise_vec[:] = np.sqrt(self.variance) * self.rng.normal(size=self.m)
                if erasure_bits is None:
                    erasure_bits = np.empty(self.m, dtype=int)
                if self.erasure_seq is not None:
                    erasure_bits[:] = self.erasure_seq[self.erasure_index, np.arange(self.m)]
                    self.erasure_index += 1
                else:
                    erasure_bits[:] = (self.rng.random(self.m) < self.eps).astype(int)
                arms_finished_this_round = set()

                # Build proposed arms first
                for m in range(self.m):
                    # 1) if this agent's assignment stack is empty, fallback:
                    if len(self.assignments[m]) == 0:
                        # pick any arm that still needs more pulls
                        still_need = [a for a in range(self.k)
                                    if self.active_arms[a] == 1
                                    and pull_counts[a] < M_i]
                        if len(still_need) == 0:
                            # fallback: pick any active arm or do nothing
                            proposed_arm = self.pulled_ind[m] if self.pulled_ind[m]>=0 else 0
                        else:
                            proposed_arm = self.rng.choice(still_need)
                    else:
                        # 2) top of the stack for agent m
                        top_arm, _ = self.assignments[m][0]
                        proposed_arm = top_arm
                    proposed_arms[m] = proposed_arm
                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms, erasure_bits)

                for m in range(self.m):
                    proposed_arm = proposed_arms[m]
                    is_erased = erasure_bits[m] if tx_mask[m] else 0
                    if not tx_mask[m]:
                        # no transmission; agent sticks with last delivered
                        actual_arm = self.pulled_ind[m] if self.pulled_ind[m] >= 0 else proposed_arm
                    else:
                        if is_erased:
                            actual_arm = self.pulled_ind[m] if self.pulled_ind[m] >= 0 else self.rng.integers(0, self.k)
                        else:
                            actual_arm = proposed_arm
                            self.pulled_ind[m] = proposed_arm

                    # 4) Reward
                    batch_agent_arm_counts[actual_arm][m] += 1
                    actual_pull_counts[actual_arm] = actual_pull_counts[actual_arm] + 1 if actual_arm in actual_pull_counts else 1
                    if self.bernoulli:
                        reward_val = float(self.rng.binomial(1, self.mu[actual_arm]))
                    else:
                        reward_val = self.mu[actual_arm] + noise_vec[m]
                    reward_vec[m] = reward_val
                    # Regret
                    regret_val = np.max(self.mu) - self.mu[actual_arm]
                    regret_vec[m] = regret_val

                    # 5) Decrement the chunk for whichever arm was actually played
                    self._decrement_chunk(m, actual_arm)

                    # 6) If the arm is active and still needs more pulls, increment effective_pulls
                    if self.active_arms[actual_arm] == 1 and pull_counts[actual_arm] < M_i:
                        pull_counts[actual_arm] += 1
                        reward_sums[actual_arm] += reward_val
                        if pull_counts[actual_arm] >= M_i:
                            arms_finished_this_round.add(actual_arm)

                    self.pulled_ind[m] = actual_arm

                for arm in arms_finished_this_round:
                    self._remove_arm_from_stacks(arm)

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


    def _decrement_chunk(self, m, arm_played):
        """
        We do 'one pull' credit for the actual arm that got played.
        So if the agent had on top-of-stack a matching arm, we decrement its chunk.
        If the top-of-stack was a different arm (the new one was erased),
        we might want to decrement the chunk for the old arm instead, if
        it still exists in the stack. This is a design choice:
          - "One chunk must be decremented each round"
          OR
          - "We only decrement the chunk for the new arm if we successfully played it."

        Here, we adopt the logic:
          If the top-of-stack arm == arm_played, decrement it.
          Otherwise, we search for arm_played in the stack, if it is there, decrement it.
        """
        if len(self.assignments[m]) == 0:
            return

        top_arm, top_count = self.assignments[m][0]
        if top_arm == arm_played:
            # We decrement from the top chunk
            top_count -= 1
            if top_count <= 0:
                # pop from the stack
                if isinstance(self.assignments[m], deque):
                    self.assignments[m].popleft()
                else:
                    self.assignments[m].pop(0)
            else:
                if isinstance(self.assignments[m], deque):
                    arm_name, _ = self.assignments[m][0]
                    self.assignments[m][0] = (arm_name, top_count)
                else:
                    self.assignments[m][0][1] = top_count

    def _remove_arm_from_stacks(self, arm):
        """
        Once an arm is done (or we do partial elimination),
        remove it from all agents' stacks so they don't keep pulling it.
        """
        for m in range(self.m):
            new_list = []
            for (a2, c2) in self.assignments[m]:
                if a2 != arm:
                    new_list.append([a2, c2])
            self.assignments[m] = deque(new_list) if isinstance(self.assignments[m], deque) else new_list

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
        self.k_reward_q = np.zeros(self.k) # Mean reward for each arm
        self.assignments = [[] for _ in range(self.m)] #assigned pulls for agents
