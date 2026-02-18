"""
BatchTPGNew — TPG variant that guards already-delivered arms from takeover,
reducing redundant retransmissions under low erasure.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.batch_tpg import BatchTPG
from models.base import C_CONFIDENCE, FEEDBACK_BEACON


class BatchTPGNew(BatchTPG):
    """TPG variant that guards already-delivered arms from reassignment.

    Agents that have successfully received their assigned arm are not candidates
    for takeover, reducing wasted retransmissions when erasure rates are low.
    """

    def __init__(self, *args, **kwargs):
        # mirror BatchTPG signature but keep a distinct display name
        kwargs = dict(kwargs)
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.name = "TPG-Guarded"

    def run(self):
        t = 0
        M_i = int(math.ceil(2 * math.log(self.iters * self.m)) / 4)

        while t < self.iters and np.any(self.active_arms == 1):
            M_i *= 4
            active_arm_inds = np.where(self.active_arms == 1)[0]
            if len(active_arm_inds) == 0:
                break

            pull_count = np.zeros(self.k, dtype=int)
            pull_sums = np.zeros(self.k)
            actual_pull_count = np.zeros(self.k, dtype=int)
            batch_agent_arm_counts = np.zeros((self.k, self.m), dtype=int)

            done_arms = set()
            done_count = 0
            done_threshold = max(0, len(active_arm_inds) - self.m)

            assigned_arm = [-1] * self.m
            taken_over = [False] * self.m
            delivery_done = [False] * self.m
            attempts_spent = [0] * self.m
            num_successes = [0] * self.m
            unassigned_arms = list(active_arm_inds)

            def compute_eta(m_idx):
                if taken_over[m_idx]:
                    return 0
                a_now = assigned_arm[m_idx]
                if a_now < 0 or a_now in done_arms:
                    return 0
                if not delivery_done[m_idx]:
                    return max(0, (self.alphas[m_idx] + (M_i - pull_count[a_now])) - attempts_spent[m_idx])
                return max(0, M_i - num_successes[m_idx])

            def pick_new_arm(m_idx):
                if len(unassigned_arms) > 0:
                    arm = unassigned_arms.pop()
                    assigned_arm[m_idx] = arm
                    delivery_done[m_idx] = False
                    attempts_spent[m_idx] = 0
                    num_successes[m_idx] = 0

            def find_slowest_agent(m_idx):
                slow_idx = -1
                slow_eta = -1
                for mm in range(self.m):
                    if taken_over[mm] or mm == m_idx:
                        continue
                    a_cur = assigned_arm[mm]
                    if a_cur < 0 or a_cur in done_arms:
                        continue
                    if delivery_done[mm] and pull_count[a_cur] < M_i:
                        continue  # protect delivered assignments
                    ceta = compute_eta(mm)
                    if ceta > slow_eta:
                        slow_eta = ceta
                        slow_idx = mm
                return slow_idx, slow_eta

            def do_takeover(m_idx):
                if assigned_arm[m_idx] != -1 or taken_over[m_idx]:
                    return
                s_idx, s_eta = find_slowest_agent(m_idx)
                if s_idx < 0 or s_idx == m_idx or s_eta <= 0:
                    return
                my_eta = (self.alphas[m_idx] + M_i)
                if my_eta < s_eta:
                    a_take = assigned_arm[s_idx]
                    if a_take < 0 or a_take in done_arms:
                        return
                    taken_over[s_idx] = True
                    assigned_arm[m_idx] = a_take
                    delivery_done[m_idx] = False
                    attempts_spent[m_idx] = 0
                    num_successes[m_idx] = 0

            while done_count < done_threshold and t < self.iters:
                for mm in range(self.m):
                    if taken_over[mm]:
                        continue
                    a_now = assigned_arm[mm]
                    if a_now < 0 or a_now in done_arms or pull_count[a_now] >= M_i:
                        assigned_arm[mm] = -1
                        pick_new_arm(mm)
                        if assigned_arm[mm] < 0:
                            do_takeover(mm)
                        if assigned_arm[mm] == -1:
                            assigned_arm[mm] = a_now
                    elif delivery_done[mm] and pull_count[a_now] < M_i:
                        # keep working on delivered arm
                        continue

                reward_vec = np.zeros(self.m)
                regret_vec = np.zeros(self.m)
                noise_vec = math.sqrt(self.variance) * self.rng.normal(size=self.m)

                proposed_arms = np.empty(self.m, dtype=int)
                for mm in range(self.m):
                    a_now = assigned_arm[mm]
                    proposed_arms[mm] = self.pulled_ind[mm] if a_now < 0 and self.pulled_ind[mm] >= 0 else (a_now if a_now >= 0 else 0)

                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms)

                for mm in range(self.m):
                    a_now = assigned_arm[mm]
                    if a_now < 0:
                        actual_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_arms[mm]
                    else:
                        if not delivery_done[mm]:
                            attempts_spent[mm] += 1
                            if attempts_spent[mm] > (self.alphas[mm] + M_i):
                                assigned_arm[mm] = -1
                                actual_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_arms[mm]
                            elif tx_mask[mm] and erasure_bits[mm] == 0:
                                actual_arm = a_now
                                self.pulled_ind[mm] = a_now
                                delivery_done[mm] = True
                                num_successes[mm] = 1
                            else:
                                actual_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_arms[mm]
                        else:
                            if tx_mask[mm] and erasure_bits[mm] == 0:
                                actual_arm = a_now
                                self.pulled_ind[mm] = a_now
                                num_successes[mm] += 1
                            else:
                                actual_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_arms[mm]

                    reward_val = self.mu[actual_arm] + noise_vec[mm]
                    reward_vec[mm] = reward_val
                    regret_vec[mm] = np.max(self.mu) - self.mu[actual_arm]
                    actual_pull_count[actual_arm] += 1
                    batch_agent_arm_counts[actual_arm][mm] += 1

                    if pull_count[actual_arm] < M_i:
                        pull_count[actual_arm] += 1
                        pull_sums[actual_arm] += reward_val
                        if pull_count[actual_arm] >= M_i:
                            done_arms.add(actual_arm)
                            done_count += 1

                self.rewards[t, :] = reward_vec
                self.regrets[t, :] = regret_vec
                self.rewards_all[t] = np.sum(reward_vec)
                self.regrets_all[t] = np.sum(regret_vec)
                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total
                t += 1
                if t >= self.iters:
                    break

            unfinished_arms = [a for a in active_arm_inds if pull_count[a] < M_i]
            if len(unfinished_arms) > 0 and len(unfinished_arms) <= self.m and t < self.iters:
                chunk_per_arm = max(1, self.m // (len(unfinished_arms)))
                needed_per_chunk = math.ceil(M_i / float(chunk_per_arm))

                replicate_list = []
                replicate_count = {}
                replicate_sum = {}

                for a in unfinished_arms:
                    for cid in range(chunk_per_arm):
                        replicate_list.append((a, cid))
                        replicate_count[(a, cid)] = 0
                        replicate_sum[(a, cid)] = 0.0
                replicate_active = {rc: True for rc in replicate_list}

                assigned_chunk = [(-1, -1)] * self.m
                chunk_taken_over = [False] * self.m

                chunk_delivery_done = [False] * self.m
                chunk_attempts_spent = [0] * self.m
                chunk_num_succ = [0] * self.m

                rep_ptr = 0

                def chunk_eta(m_idx):
                    if chunk_taken_over[m_idx]:
                        return 0
                    (rA, rC) = assigned_chunk[m_idx]
                    if rA < 0 or not replicate_active.get((rA, rC), False):
                        return 0
                    if replicate_count.get((rA, rC), 0) >= needed_per_chunk:
                        return 0
                    if not chunk_delivery_done[m_idx]:
                        return max(0, (self.alphas[m_idx] + (needed_per_chunk - pull_count[rA])) - chunk_attempts_spent[m_idx])
                    return max(0, needed_per_chunk - chunk_num_succ[m_idx])

                def find_slowest_chunk_agent(m_idx):
                    slow_idx = -1
                    slow_eta = -1
                    for mm in range(self.m):
                        if mm == m_idx or chunk_taken_over[mm]:
                            continue
                        (ra, rcid) = assigned_chunk[mm]
                        if ra < 0 or rcid < 0:
                            continue
                        if not replicate_active.get((ra, rcid), False):
                            continue
                        if chunk_delivery_done[mm] and pull_count[ra] < M_i:
                            continue
                        ceta = chunk_eta(mm)
                        if ceta > slow_eta:
                            slow_eta = ceta
                            slow_idx = mm
                    return slow_idx, slow_eta

                def chunk_takeover(m_idx):
                    if chunk_taken_over[m_idx]:
                        return
                    s_idx, s_eta = find_slowest_chunk_agent(m_idx)
                    if s_idx < 0 or s_idx == m_idx or s_eta <= 0:
                        return
                    my_eta = (self.alphas[m_idx] + (needed_per_chunk - pull_count[assigned_chunk[s_idx][0]]))
                    if my_eta < s_eta:
                        (sA, sC) = assigned_chunk[s_idx]
                        if sA < 0 or sA in done_arms:
                            return
                        chunk_taken_over[s_idx] = True
                        assigned_chunk[m_idx] = (sA, sC)
                        chunk_delivery_done[m_idx] = False
                        chunk_attempts_spent[m_idx] = 0
                        chunk_num_succ[m_idx] = 0

                def chunk_done(a, cid):
                    if not replicate_active.get((a, cid), False):
                        return True
                    if replicate_count[(a, cid)] >= needed_per_chunk:
                        return True
                    if pull_count[a] >= M_i:
                        return True
                    return False

                def remove_all_copies(a):
                    for (xx, cc) in replicate_list:
                        if xx == a:
                            replicate_active[(xx, cc)] = False

                limit_ass = min(self.m, len(replicate_list))
                for iA in range(self.m):
                    if iA < len(replicate_list):
                        assigned_chunk[iA] = replicate_list[iA]
                    else:
                        random_arm = self.rng.choice(unfinished_arms)
                        assigned_chunk[iA] = (random_arm, -1)
                rep_ptr = limit_ass

                def any_active_chunks():
                    for (aa, cc) in replicate_list:
                        if replicate_active.get((aa, cc), False) and replicate_count[(aa, cc)] < needed_per_chunk and pull_count[aa] < M_i:
                            return True
                    return False
                if self.verbose:
                    self.logger.debug('second phase start')
                while any_active_chunks() and t < self.iters:
                    reward_vec = np.zeros(self.m)
                    regret_vec = np.zeros(self.m)

                    for mm in reversed(range(self.m)):
                        if chunk_taken_over[mm]:
                            continue
                        (ra, rcid) = assigned_chunk[mm]
                        if ra < 0 or chunk_done(ra, rcid):
                            while rep_ptr < len(replicate_list):
                                cand = replicate_list[rep_ptr]
                                rep_ptr += 1
                                if not chunk_done(*cand):
                                    assigned_chunk[mm] = cand
                                    (ra, rcid) = cand
                                    break
                            else:
                                chunk_takeover(mm)
                            if assigned_chunk[mm][0] < 0:
                                assigned_chunk[mm] = (ra, rcid)

                    if self.verbose:
                        self.logger.debug('assigned_chunk=%s', assigned_chunk)
                    noise_vec = math.sqrt(self.variance) * self.rng.normal(size=self.m)

                    proposed_chunk = np.empty(self.m, dtype=int)
                    for mm in range(self.m):
                        (ra, rcid) = assigned_chunk[mm]
                        if ra < 0:
                            proposed_chunk[mm] = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else 0
                        else:
                            proposed_chunk[mm] = ra
                    tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_chunk)

                    for mm in range(self.m):
                        (ra, rcid) = assigned_chunk[mm]
                        if ra < 0:
                            act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                        else:
                            if not chunk_delivery_done[mm]:
                                chunk_attempts_spent[mm] += 1
                                budget = (self.alphas[mm] + needed_per_chunk)
                                if chunk_attempts_spent[mm] > budget:
                                    assigned_chunk[mm] = (-1, -1)
                                    act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                                elif tx_mask[mm] and erasure_bits[mm] == 0:
                                    act_arm = ra
                                    self.pulled_ind[mm] = ra
                                    chunk_delivery_done[mm] = True
                                    chunk_num_succ[mm] = 1
                                else:
                                    act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                            else:
                                if tx_mask[mm] and erasure_bits[mm] == 0:
                                    act_arm = ra
                                    self.pulled_ind[mm] = ra
                                    chunk_num_succ[mm] += 1
                                else:
                                    act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]

                        reward_val = self.mu[act_arm] + noise_vec[mm]
                        reward_vec[mm] = reward_val
                        regret_vec[mm] = np.max(self.mu) - self.mu[act_arm]
                        actual_pull_count[act_arm] += 1
                        batch_agent_arm_counts[act_arm][mm] += 1

                        if pull_count[act_arm] < M_i:
                            pull_count[act_arm] += 1
                            pull_sums[act_arm] += reward_val
                            if pull_count[act_arm] >= M_i:
                                remove_all_copies(act_arm)

                        if act_arm == ra and ra >= 0 and replicate_active.get((ra, rcid), False):
                            replicate_count[(ra, rcid)] += 1
                            if replicate_count[(ra, rcid)] >= needed_per_chunk or pull_count[ra] >= M_i:
                                replicate_active[(ra, rcid)] = False

                    self.rewards[t, :] = reward_vec
                    self.regrets[t, :] = regret_vec
                    self.rewards_all[t] = np.sum(reward_vec)
                    self.regrets_all[t] = np.sum(regret_vec)
                    self.tx_over_time[t] = self.tx_count_total
                    self.fb_over_time[t] = self.fb_count_total
                    t += 1
                    if t >= self.iters:
                        break

            for a in active_arm_inds:
                if pull_count[a] > 0:
                    self.arm_sums[a] += pull_sums[a]
                    self.arm_counts[a] += pull_count[a]
                    self.k_reward_q[a] = self.arm_sums[a] / float(self.arm_counts[a])

            survivors = np.where(self.active_arms == 1)[0]
            if len(survivors) == 0:
                break
            best_mean = np.max(self.k_reward_q[survivors])
            for a in survivors:
                diff = best_mean - self.k_reward_q[a]
                threshold = C_CONFIDENCE * self.c * np.sqrt(np.log(self.iters * self.m) / (2.0 * M_i))
                if diff > threshold:
                    self.active_arms[a] = 0
