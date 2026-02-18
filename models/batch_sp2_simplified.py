"""
BatchSP2Simplified — Simplified TPG using pointer-based scheduling. Uses a
single pointer to cycle agents through arms, with chunk-based finishing when
few arms remain.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, FEEDBACK_BEACON


class BatchSP2Simplified(BanditBase):
    """Simplified TPG using pointer-based scheduling.

    Tracks per-arm pull counts and reward sums locally within each batch, then
    folds into global statistics for elimination. Uses a worst-channel pointer
    to recycle agents through arms.
    """

    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_BEACON, rng: Optional[Generator] = None, verbose: bool = False):
        super().__init__(
            name='TPG-Simplified',
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

        # Empirical means of each arm, plus global counters:
        self.k_reward_q = np.zeros(k)
        self.arm_counts = np.zeros(k, dtype=int)  # total number of successful pulls so far
        self.arm_sums = np.zeros(k)               # total reward accumulated



    def run(self):
        '''
        Core logic of pointer-based scheduling + batch elimination in a single function.

        - We create local pull_count[a] and pull_sums[a] at each batch,
          and update them in real time whenever an arm is played (not erased).
        - At the end of the batch, we fold these local sums into global sums
          (self.arm_sums[a], self.arm_counts[a]) to get an accurate empirical mean.
        '''
        t = 0  # global round counter
        #batch_index = 0
        M_i = int(np.ceil(2 * np.log(self.iters * self.m )) / 4)

        # As long as we have time left and arms to sample:
        while t < self.iters and np.any(self.active_arms == 1):
            # Multiply required pulls by 4 each batch
            #batch_index += 1
            M_i *= 4 #4**batch_index
            active_arm_inds = np.where(self.active_arms == 1)[0]
            if len(active_arm_inds) == 0:
                break

            # Local counters for this batch
            active_mask = np.zeros(self.k, dtype=bool)
            active_mask[active_arm_inds] = True
            pull_count = np.zeros(self.k, dtype=int)
            pull_sums = np.zeros(self.k)
            actual_pull_count = np.zeros(self.k, dtype=int)


            # Agents' assigned arms (-1 => no assigned arm yet)
            assigned_arm = [-1]*self.m
            # Right pointer (worst channel index)
            pt = self.m - 1

            # Shuffle the active arms once per batch
            #random.shuffle(active_arm_inds)

            # We'll treat the first min(m, #active) arms as "scheduled" initially
            scheduled_arms = set()
            unsched_ptr = 0  # points to the next unscheduled arm index in active_arm_inds

            limit_assign = min(self.m, len(active_arm_inds))
            for i in range(limit_assign):
                # Assign to the best channels first
                assigned_arm[i] = active_arm_inds[i]
                scheduled_arms.add(active_arm_inds[i])
            unsched_ptr += limit_assign

            def unfinished_arms():
                """Return list of arms that haven't yet reached M_i effective pulls."""
                mask = active_mask & (pull_count < M_i)
                return np.flatnonzero(mask)

            batch_agent_arm_counts = np.zeros((self.k, self.m), dtype=int)
            # 1) MAIN LOOP for large-batch scenario: while #unfinished > m, run round by round
            while len(unfinished_arms()) > self.m and t < self.iters:
                reward_vec = np.zeros(self.m)
                regret_vec = np.zeros(self.m)
                noise_vec = np.sqrt(self.variance) * self.rng.normal(size=self.m)

                # We go from worst agent (m-1) up to best (0).
                for m_idx in range(self.m):
                    candidate_arm = assigned_arm[m_idx]
                    if candidate_arm < 0 or (candidate_arm >= 0 and pull_count[candidate_arm] >= M_i):
                        if unsched_ptr < len(active_arm_inds):
                            candidate_arm = active_arm_inds[unsched_ptr]
                            assigned_arm[m_idx] = candidate_arm
                            scheduled_arms.add(candidate_arm)
                            unsched_ptr += 1
                        else:
                            if m_idx < pt:
                                assigned_arm[m_idx] = assigned_arm[pt]
                                pt = max(pt-1, 0)
                            #candidate_arm = self.pulled_ind[m_idx]

                proposed_arms = np.array(assigned_arm, dtype=int)
                tx_mask, erasure_bits_round = self.sample_erasure_and_update_comm(proposed_arms)
                for m_idx in range(self.m):
                    candidate_arm = assigned_arm[m_idx]
                    if not tx_mask[m_idx]:
                        actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else candidate_arm
                    elif erasure_bits_round[m_idx]:
                        actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else candidate_arm
                    else:
                        actual_arm = candidate_arm
                        self.pulled_ind[m_idx] = candidate_arm
                    # Reward
                    batch_agent_arm_counts[actual_arm][m_idx] += 1
                    actual_pull_count[actual_arm] += 1
                    r_val = self.mu[actual_arm] + noise_vec[m_idx]
                    reward_vec[m_idx] = r_val
                    regret_vec[m_idx] = np.max(self.mu) - self.mu[actual_arm]

                    # If this arm is still in the batch's scope (active and not done)
                    if active_mask[actual_arm] and pull_count[actual_arm] < M_i:
                        pull_count[actual_arm] += 1
                        pull_sums[actual_arm] += r_val

                # Logging
                self.rewards[t, :] = reward_vec
                self.regrets[t, :] = regret_vec
                self.rewards_all[t] = np.sum(reward_vec)
                self.regrets_all[t] = np.sum(regret_vec)

                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total

                t += 1
                if t >= self.iters:
                    break

            # Suppose we are in the "small set finishing" phase. We have:
            #   - active_arm_inds = the arms still active in this batch
            #   - pull_count[a], pull_sums[a] the local counters for the batch for each active arm a
            #   - M_i = 4^i  (the total required #pulls for each arm in this batch)
            #   - pointer-based data structures: assigned_arm[m], pt, etc.
            #   - t < self.iters is the global time counter

            unfinished_arms_lst = list(unfinished_arms())
            Khat = len(unfinished_arms_lst)

            if Khat == 0:
                # no arms to do => we are done
                pass
            else:
                # 1) define chunk_per_arm, needed_per_chunk
                chunk_per_arm = max(1, self.m // ( Khat))  # or (2*Khat) if you prefer
                needed_per_chunk = math.ceil(M_i / chunk_per_arm)

                # 2) build replicate_list, replicate_count, replicate_sum
                replicate_list = [(a, cid) for a in unfinished_arms_lst for cid in range(chunk_per_arm)]
                replicate_count = {rc: 0 for rc in replicate_list}
                replicate_sum = {rc: 0.0 for rc in replicate_list}
                # dictionary to see if a chunk is still "active" (not finished & not removed)
                replicate_active = {rc: True for rc in replicate_list}

                # agent assignments: each agent has assigned_arm[m] = (a,cid) or (-1,-1)
                assigned_arm = [(-1, -1)] * self.m
                # pointer to unscheduled replicate copies
                rep_unsched_ptr = 0

                # ------------------------------------------------------------------------
                # MODIFICATION (A): Assign replicate copies to some agents,
                # and then fill leftover agents with a random unfinished arm.
                # ------------------------------------------------------------------------
                limit_assign = min(self.m, len(replicate_list))
                last_assigned_agent = -1

                # 1) Assign replicate_list copies to agents from best->worst
                for i in range(limit_assign):
                    assigned_arm[i] = replicate_list[i]
                    rep_unsched_ptr += 1
                    last_assigned_agent = i

                # 2) If leftover agents remain, assign them a random active arm
                if last_assigned_agent < self.m - 1:
                    for j in range(last_assigned_agent+1, self.m):
                        # pick random from the currently unfinished arms
                        random_arm = self.rng.choice(unfinished_arms_lst)
                        assigned_arm[j] = (random_arm, -1)
                # Place the pointer 'pt' at the last agent who actually got a meaningful assignment
                pt = last_assigned_agent
                # ------------------------------------------------------------------------

                def chunk_is_done(a, cid):
                    """Returns True if that chunk is finished or if the arm as a whole is done."""
                    if not replicate_active.get((a,cid), False):
                        return True
                    if replicate_count[(a,cid)] >= needed_per_chunk:
                        return True
                    # also if the arm as a whole is done
                    if pull_count[a] >= M_i:
                        return True
                    return False

                def remove_all_copies_of_arm(a):
                    """If arm a is globally done, remove all chunk copies from replicate_active."""
                    for (arm_x, cidx_x) in replicate_list:
                        if arm_x == a:
                            replicate_active[(arm_x,cidx_x)] = False

                def unfinished_copies_exist():
                    """Check if there is any chunk copy still active and not done."""
                    for (a,cid) in replicate_list:
                        if replicate_active[(a,cid)] and not chunk_is_done(a,cid):
                            return True
                    return False

            # 3) single pointer-based loop (reuse buffers)
            reward_vec = np.zeros(self.m)
            regret_vec = np.zeros(self.m)
            noise_vec = np.empty(self.m)
            while unfinished_copies_exist() and t < self.iters:
                noise_vec[:] = np.sqrt(self.variance) * self.rng.normal(size=self.m)

                # iterate agents from worst to best
                for m_idx in reversed(range(self.m)):
                    (ra, rcid) = assigned_arm[m_idx]
                    # if the chunk is done or invalid, pick a new chunk
                    if (ra < 0 or chunk_is_done(ra, rcid)):
                        # attempt to pick unscheduled replicate
                        newly_assigned = False
                        while rep_unsched_ptr < len(replicate_list):
                            candidate = replicate_list[rep_unsched_ptr]
                            rep_unsched_ptr += 1
                            if not chunk_is_done(*candidate):
                                assigned_arm[m_idx] = candidate
                                (ra, rcid) = candidate
                                break
                        if not newly_assigned:
                            if m_idx < pt:
                                assigned_arm[m_idx] = assigned_arm[pt]
                                pt = max(pt - 1, 0)
                        #else:
                            # fallback => no candidate chunk => agent keeps last pulled arm
                        #    ra, rcid = -1, -1
                proposed_arms = np.zeros(self.m, dtype=int)
                for m_idx in range(self.m):
                    ra, _ = assigned_arm[m_idx]
                    if ra >= 0:
                        proposed_arms[m_idx] = ra
                    else:
                        proposed_arms[m_idx] = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else 0

                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms)

                for m_idx in range(self.m):
                    (ra, rcid) = assigned_arm[m_idx]
                    candidate_arm = proposed_arms[m_idx]

                    if not tx_mask[m_idx]:
                        actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else candidate_arm
                    elif erasure_bits[m_idx]:
                        actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else candidate_arm
                    else:
                        actual_arm = candidate_arm
                        self.pulled_ind[m_idx] = candidate_arm

                    # Reward
                    batch_agent_arm_counts[actual_arm][m_idx] += 1
                    actual_pull_count[actual_arm] += 1
                    r_val = self.mu[actual_arm] + noise_vec[m_idx]
                    reward_vec[m_idx] = r_val
                    regret_vec[m_idx] = np.max(self.mu) - self.mu[actual_arm]

                    # If this chunk copy is active, increment replicate_count
                    if actual_arm == ra and ra >= 0 and replicate_active.get((ra, rcid), False):
                        # increment chunk
                        replicate_count[(ra, rcid)] += 1
                        replicate_sum[(ra, rcid)] += r_val
                        # also update real arm
                        pull_count[ra] += 1
                        pull_sums[ra] += r_val

                        # check if arm ra is done overall
                        if pull_count[ra] >= M_i:
                            # remove all copies
                            remove_all_copies_of_arm(ra)
                        elif replicate_count[(ra, rcid)] >= needed_per_chunk: # or check if chunk is done
                            # free agent
                            replicate_active[(ra, rcid)] = False
                    elif actual_arm in unfinished_arms_lst:
                        pull_count[actual_arm] += 1
                        pull_sums[actual_arm] += r_val

                        # check if arm ra is done overall
                        #if pull_count[actual_arm] >= M_i:
                            # remove all copies
                        #    remove_all_copies_of_arm(actual_arm)

                # Logging each round
                self.rewards[t, :] = reward_vec
                self.regrets[t, :] = regret_vec
                self.rewards_all[t] = np.sum(reward_vec)
                self.regrets_all[t] = np.sum(regret_vec)

                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total

                t += 1


            # 3) End of batch => fold local pull_sums/cnts into global, then eliminate arms
            for a in active_arm_inds:
                local_cnt = pull_count[a]
                local_sum = pull_sums[a]
                if local_cnt > 0:
                    self.arm_sums[a] += local_sum
                    self.arm_counts[a] += local_cnt
                    self.k_reward_q[a] = self.arm_sums[a] / float(self.arm_counts[a])

            self.eliminate_arms(M_i)


    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        '''
        Reset logs and environment parameters while keeping the fundamental settings.
        '''
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k, dtype=int)
        self.arm_sums = np.zeros(self.k)
