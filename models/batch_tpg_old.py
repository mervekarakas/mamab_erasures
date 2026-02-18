"""
BatchTPGOld — Earlier TPG variant with forcible agent unassignment.
Superseded by BatchTPG.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_BEACON


class BatchTPGOld(BanditBase):
    """
    A batched multi-agent MAB algorithm with erasure feedback,
    using a Two-Phase Greedy (TPG) approach + chunk-based finishing.

    Main Changes from the pointer-based approach:
      (1) 'Phase 1': Greedy assignment with "takeover":
          - Agents repeatedly pick up arms that need M_i=4^i successes.
          - If an agent finishes early or is idle, it can 'take over'
            from a slow agent to reduce overall finishing time.

      (2) 'Phase 2': If <= M arms remain unfinished, chunk them so
          multiple agents can share them more effectively.

    We keep local pull_count[a] & pull_sums[a] for each batch, then fold
    them into self.arm_counts[a], self.arm_sums[a] at batch-end. Then partial-eliminate
    suboptimal arms from self.active_arms using confidence intervals.
    """

    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0,
                 base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_BEACON, rng: Optional[Generator] = None, verbose: bool = False):
        # Basic settings routed through BanditBase
        super().__init__(
            name='TPG-Old',
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

        self.k_reward_q = np.zeros(k)            # current empirical mean of each arm
        self.arm_counts = np.zeros(k, dtype=int) # total # of successful pulls so far
        self.arm_sums = np.zeros(k)              # total reward accumulated so far

    def _get_erasure_bit(self, m_idx: int) -> int:
        """
        Return the next erasure bit from the predetermined sequence if available;
        otherwise, sample from Bernoulli(eps[m_idx]).
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
        Main Two-Phase Greedy loop over batches. At each batch i:
          (1) We require M_i ~ 4^i successful pulls for each active arm.
          (2) Phase 1: Greedy assignment with 'takeover' logic until (K - M)+ arms done.
          (3) Phase 2: If leftover arms <= M, chunk them & do multi-agent finishing.
          (4) Merge local counters into global arm_counts & arm_sums, eliminate suboptimal arms.
        """
        t = 0  # global time
        # We start with some initial M_i
        M_i = int(np.ceil(2 * np.log(self.iters * self.m )) / 4)

        # As long as we have time left and arms remain:
        while t < self.iters and np.any(self.active_arms == 1):
            M_i *= 4
            active_arm_inds = np.where(self.active_arms == 1)[0]
            if len(active_arm_inds) == 0:
                break

            # local counters
            pull_count = {a: 0 for a in active_arm_inds}
            pull_sums = {a: 0.0 for a in active_arm_inds}
            actual_pull_count = {a: 0 for a in active_arm_inds}  # debugging

            # ======== PHASE 1: GREEDY TAKEOVER =============
            done_arms = set()  # arms that already reached M_i
            # Each agent is assigned either an arm or idle (-1)
            assigned_arm = [-1]*self.m
            # For each agent, track how many attempts are used so far for the current arm
            attempt_used = np.zeros(self.m, dtype=int)

            # Unassigned arms
            unassigned_arms = list(active_arm_inds)  # we can shuffle if we want
            # keep track of how many are completed
            done_count = 0
            done_threshold = max(0, len(active_arm_inds)-self.m)

            def get_estimated_end_time(m_idx):
                """Compute worst-case leftover for agent m_idx's current arm."""
                a_now = assigned_arm[m_idx]
                if a_now == -1 or a_now in done_arms:
                    return 0
                attempts_spent = attempt_used[m_idx]
                # worst-case budget for this batch is alpha_m + M_i
                budget = self.alphas[m_idx] + M_i
                leftover = max(0, budget - attempts_spent)
                return leftover

            # Utility: pick new arm if agent is idle
            def pick_new_arm(m_idx):
                nonlocal unassigned_arms
                if len(unassigned_arms) == 0:
                    return
                a = unassigned_arms.pop()
                assigned_arm[m_idx] = a
                attempt_used[m_idx] = 0

            # Utility: takeover from slowest agent
            def do_takeover(m_idx):
                """If agent m_idx is idle, find the slowest agent that is still working,
                   if that reduces overall finish time."""
                if assigned_arm[m_idx] != -1:
                    return  # not idle
                # find slowest agent
                slow_idx = None
                slow_eta = -1
                for mm in range(self.m):
                    if mm == m_idx:
                        continue
                    if assigned_arm[mm] == -1 or assigned_arm[mm] in done_arms:
                        continue
                    eta_mm = get_estimated_end_time(mm)
                    if eta_mm > slow_eta:
                        slow_eta = eta_mm
                        slow_idx = mm
                if slow_idx is None or slow_eta <= 0:
                    return
                # Attempt a takeover if it helps
                # We'll do a naive approach: agent m_idx just picks the same arm from slow_idx
                # "Takeover" only if leftover for m_idx < leftover for slow_idx
                # But we need some logic to estimate if it helps. We'll do a simpler approach:
                # if idle agent is "faster"? Actually, all have same attempts_spent logic,
                # so let's just do a direct takeover
                a_take = assigned_arm[slow_idx]
                if a_take < 0 or a_take in done_arms:
                    return
                assigned_arm[m_idx] = a_take
                assigned_arm[slow_idx] = -1
                used_prev = attempt_used[slow_idx]
                attempt_used[slow_idx] = 0
                attempt_used[m_idx] = used_prev

            # Main loop of Phase 1
            while done_count < done_threshold and t < self.iters:
                # Step 1: if agent is idle or done, pick new arm or takeover
                for m_idx in range(self.m):
                    a_now = assigned_arm[m_idx]
                    if a_now == -1 or a_now in done_arms or pull_count[a_now]>=M_i:
                        # pick new or try takeover
                        assigned_arm[m_idx] = -1
                        pick_new_arm(m_idx)
                        if assigned_arm[m_idx] == -1:
                            # try takeover
                            do_takeover(m_idx)

                # Step 2: build reward/regret logs
                reward_vec = np.zeros(self.m)
                regret_vec = np.zeros(self.m)
                if not self.bernoulli:
                    noise_vec = np.sqrt(self.variance) * self.rng.normal(size=self.m)

                # Step 3: for each agent, do one attempt
                # Build proposed arms for this round
                proposed_arms = np.empty(self.m, dtype=int)
                for m_idx in range(self.m):
                    a_now = assigned_arm[m_idx]
                    if a_now == -1 or a_now in done_arms:
                        proposed_arms[m_idx] = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else 0
                    else:
                        proposed_arms[m_idx] = a_now

                tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_arms)

                for m_idx in range(self.m):
                    a_now = proposed_arms[m_idx]
                    if not tx_mask[m_idx]:
                        actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else a_now
                    else:
                        attempt_used[m_idx] += 1
                        if attempt_used[m_idx] > (self.alphas[m_idx] + M_i):
                            assigned_arm[m_idx] = -1
                            actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else a_now
                        else:
                            ebit = erasure_bits[m_idx]
                            if ebit:
                                actual_arm = self.pulled_ind[m_idx] if self.pulled_ind[m_idx] >= 0 else a_now
                            else:
                                actual_arm = a_now
                                self.pulled_ind[m_idx] = a_now
                    # reward
                    if self.bernoulli:
                        r_val = float(self.rng.binomial(1, self.mu[actual_arm]))
                    else:
                        r_val = self.mu[actual_arm] + noise_vec[m_idx]
                    reward_vec[m_idx] = r_val
                    regret_vec[m_idx] = np.max(self.mu)-self.mu[actual_arm]

                    # if actual_arm is among active
                    if pull_count[actual_arm]<M_i:
                        pull_count[actual_arm]+=1
                        pull_sums[actual_arm]+=r_val
                        actual_pull_count[actual_arm]+=1
                        # check if it's done
                        if pull_count[actual_arm]>=M_i:
                            done_arms.add(actual_arm)
                            done_count+=1

                # Logging
                self.rewards[t,:]=reward_vec
                self.regrets[t,:]=regret_vec
                self.rewards_all[t]=np.sum(reward_vec)
                self.regrets_all[t]=np.sum(regret_vec)

                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total


                t+=1
                if t>=self.iters:
                    break

            # ======== PHASE 2: if leftover arms <= m, chunk them =============
            unfinished_arms = [a for a in active_arm_inds if pull_count[a]<M_i]
            if len(unfinished_arms)>0 and len(unfinished_arms)<=self.m and t<self.iters:
                # chunk approach
                chunk_per_arm = max(1, self.m//(len(unfinished_arms)))
                needed_per_chunk = math.ceil(M_i/chunk_per_arm)
                # Build replicate_list
                replicate_list=[]
                replicate_count={}
                replicate_sum={}
                for a in unfinished_arms:
                    for cid in range(chunk_per_arm):
                        replicate_list.append((a,cid))
                        replicate_count[(a,cid)]=0
                        replicate_sum[(a,cid)]=0.0
                replicate_active={rc:True for rc in replicate_list}
                # agent assignment
                assigned_chunk=[(-1,-1)]*self.m
                rep_idx=0

                def chunk_done(a,cid):
                    if not replicate_active.get((a,cid),False):
                        return True
                    if replicate_count[(a,cid)]>=needed_per_chunk:
                        return True
                    if pull_count[a]>=M_i:
                        return True
                    return False
                def remove_all_copies(a):
                    for (x,c) in replicate_list:
                        if x==a:
                            replicate_active[(x,c)]=False

                # init
                limit_ass=min(self.m,len(replicate_list))
                for iA in range(limit_ass):
                    assigned_chunk[iA]=replicate_list[iA]
                rep_idx=limit_ass
                pt=limit_ass-1

                def any_active_chunks():
                    for (a,cid) in replicate_list:
                        if replicate_active[(a,cid)] and not chunk_done(a,cid):
                            return True
                    return False

                while any_active_chunks() and t<self.iters:
                    reward_vec=np.zeros(self.m)
                    regret_vec=np.zeros(self.m)
                    # try reassign from worst to best
                    for mm in reversed(range(self.m)):
                        (ra,rcid)=assigned_chunk[mm]
                        if ra<0 or chunk_done(ra,rcid):
                            # pick new replicate
                            newly_assigned=False
                            while rep_idx<len(replicate_list):
                                cand=replicate_list[rep_idx]
                                rep_idx+=1
                                if not chunk_done(*cand):
                                    assigned_chunk[mm]=cand
                                    (ra,rcid)=cand
                                    break
                            # fallback => if still no chunk, swap with best pointer
                            if not newly_assigned:
                                if mm<pt:
                                    assigned_chunk[mm]=assigned_chunk[pt]
                                    pt=max(pt-1,0)
                    # do attempt
                    # build proposed arms for chunk step
                    proposed_chunk = np.empty(self.m, dtype=int)
                    for mm in range(self.m):
                        (ra, rcid) = assigned_chunk[mm]
                        if ra < 0:
                            proposed_chunk[mm] = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else 0
                        else:
                            proposed_chunk[mm] = ra
                    tx_mask, erasure_bits = self.sample_erasure_and_update_comm(proposed_chunk)

                    for mm in range(self.m):
                        (ra,rcid)=assigned_chunk[mm]
                        if not tx_mask[mm]:
                            act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                        else:
                            ebit = erasure_bits[mm]
                            if ra<0:
                                act_arm = self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                            else:
                                if ebit:
                                    act_arm=self.pulled_ind[mm] if self.pulled_ind[mm] >=0 else proposed_chunk[mm]
                                else:
                                    act_arm=ra
                                    self.pulled_ind[mm]=ra
                        # reward
                        noise=np.sqrt(self.variance)*self.rng.normal()
                        r_val=self.mu[act_arm] + noise
                        reward_vec[mm]=r_val
                        regret_vec[mm]=np.max(self.mu)-self.mu[act_arm]
                        if pull_count[act_arm]<M_i:
                            pull_count[act_arm]+=1
                            pull_sums[act_arm]+=r_val
                            if pull_count[act_arm]>=M_i:
                                remove_all_copies(act_arm)
                        if ra>=0 and replicate_active.get((ra,rcid),False):
                            replicate_count[(ra,rcid)]+=1
                            replicate_sum[(ra,rcid)]+=r_val
                            if replicate_count[(ra,rcid)]>=needed_per_chunk or pull_count[ra]>=M_i:
                                replicate_active[(ra,rcid)]=False
                                # free agent
                    # logging
                    self.rewards[t,:]=reward_vec
                    self.regrets[t,:]=regret_vec
                    self.rewards_all[t]=np.sum(reward_vec)
                    self.regrets_all[t]=np.sum(regret_vec)

                    self.tx_over_time[t] = self.tx_count_total
                    self.fb_over_time[t] = self.fb_count_total

                    t+=1
                    if t>=self.iters:
                        break
                # end chunk approach
            # end Phase2

            # 3) End of batch => fold local sums
            for a in active_arm_inds:
                if pull_count[a]>0:
                    self.arm_sums[a]+=pull_sums[a]
                    self.arm_counts[a]+=pull_count[a]
                    self.k_reward_q[a]=self.arm_sums[a]/float(self.arm_counts[a])

            # partial elimination
            survivors=np.where(self.active_arms==1)[0]
            if len(survivors)==0:
                break
            best_mean=np.max(self.k_reward_q[survivors])
            for a in survivors:
                diff=best_mean-self.k_reward_q[a]
                threshold = C_CONFIDENCE*self.c*np.sqrt(np.log(self.iters*self.m)/(2.0*M_i))
                if diff>threshold:
                    self.active_arms[a]=0

                if self.verbose:
                    self.logger.debug('end_time=%s pull_cnt=%s survivors=%s',
                                      t,
                                      {a: actual_pull_count[a] for a in active_arm_inds},
                                      np.where(self.active_arms==1)[0])

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        """
        Reset logs and environment parameters while keeping the fundamental settings.
        """
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)
        self.k_reward_q = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k, dtype=int)
        self.arm_sums = np.zeros(self.k)
