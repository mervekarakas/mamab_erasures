"""
BatchTPG — Two-Phase Greedy, the main algorithm from JSAIT 2026.
Uses SAE + stop-on-success + dynamic takeover without forcible unassignment.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from models.base import BanditBase, C_CONFIDENCE, FEEDBACK_BEACON


class BatchTPG(BanditBase):
    """Two-Phase Greedy (TPG) for multi-agent bandits over erasure channels.

    Phase 1: greedy assignment with dynamic takeover — idle agents claim arms from
    slow agents (marked taken_over, not forcibly unassigned). No partial-state copy.
    Phase 2: chunk-based finishing when <=M arms remain.
    """

    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random',
                 epsilon=0, base=None, erasure_seq=None,
                 feedback_mode=FEEDBACK_BEACON, rng: Optional[Generator] = None, verbose: bool = False):
        super().__init__(
            name='TPG',
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
        self.arm_counts = np.zeros(k, dtype=int)
        self.arm_sums = np.zeros(k)

    def run(self):
        """
        Main TPG loop over batches. Each batch i:
          - Each active arm needs M_i=4^i successful pulls.
          - Phase1: 'Greedy' assignment, leftover-based logic, no partial-state copying.
                   We do not forcibly set slow agent to -1 but set taken_over= True.
          - Phase2: chunk leftover arms if <= M remain, same logic.
          - Merge local sums/cnts, partial elimination.
        """
        t = 0
        M_i = int(math.ceil(2 * math.log(self.iters * self.m)) / 4)

        while t < self.iters and np.any(self.active_arms == 1):
            M_i *= 4
            active_arm_inds = np.where(self.active_arms == 1)[0]
            if len(active_arm_inds) == 0:
                break

            # local counters for this batch
            pull_count = np.zeros(self.k, dtype=int)
            pull_sums = np.zeros(self.k)
            actual_pull_count = np.zeros(self.k, dtype=int)

            batch_agent_arm_counts = np.zeros((self.k, self.m), dtype=int)

            # ======== PHASE 1: GREEDY TAKEOVER (no partial-state copy) =============
            done_arms = set()
            done_count = 0
            done_threshold = max(0, len(active_arm_inds) - self.m)

            assigned_arm = [-1]*self.m
            taken_over  = [False]*self.m   # new flag

            # for leftover logic, we do a simple model:
            # if delivery_done[m] = False => leftover ~ alpha_m[m] + M_i - attempts_spent[m]
            # else => leftover ~ M_i - num_successes[m]
            delivery_done = [False]*self.m
            attempts_spent= [0]*self.m
            num_successes= [0]*self.m

            unassigned_arms = list(active_arm_inds)

            def compute_eta(m_idx):
                # If agent is taken_over => skip
                if taken_over[m_idx]:
                    return 0
                a_now = assigned_arm[m_idx]
                if a_now<0 or a_now in done_arms:
                    return 0
                if not delivery_done[m_idx]:
                    return max(0,(self.alphas[m_idx]+(M_i - pull_count[a_now]))-attempts_spent[m_idx])
                else:
                    return max(0,M_i - num_successes[m_idx])

            def pick_new_arm(m_idx):
                if len(unassigned_arms)>0:
                    arm=unassigned_arms.pop()
                    assigned_arm[m_idx]=arm
                    delivery_done[m_idx]=False
                    attempts_spent[m_idx]=0
                    num_successes[m_idx]=0

            def find_slowest_agent(m_idx):
                slow_idx=-1
                slow_eta=-1
                for mm in range(self.m):
                    if taken_over[mm]:
                        continue
                    if mm == m_idx:
                        continue
                    a_cur=assigned_arm[mm]
                    if a_cur<0 or a_cur in done_arms:
                        continue
                    ceta=compute_eta(mm)
                    if ceta>slow_eta:
                        slow_eta=ceta
                        slow_idx=mm
                return slow_idx,slow_eta

            def do_takeover(m_idx):
                # if agent is not idle or if it's taken over itself, skip
                if assigned_arm[m_idx]!=-1 or taken_over[m_idx]:
                    return
                s_idx,s_eta=find_slowest_agent(m_idx)
                if s_idx<0 or s_idx==m_idx or s_eta<=0:
                    return
                # leftover if we start from scratch
                my_eta=(self.alphas[m_idx]+M_i)
                if my_eta<s_eta:
                    a_take=assigned_arm[s_idx]
                    if a_take<0 or a_take in done_arms:
                        return
                    # we keep the slow agent's arm assignment but set taken_over
                    taken_over[s_idx]=True
                    # new agent assigned, but no partial copy:
                    assigned_arm[m_idx]=a_take
                    # do NOT copy partial progress
                    delivery_done[m_idx]=False
                    attempts_spent[m_idx]=0
                    num_successes[m_idx]=0

            # main loop
            while done_count<done_threshold and t<self.iters:
                # fill idle/done
                for mm in range(self.m):
                    if taken_over[mm]:
                        continue
                    a_now=assigned_arm[mm]
                    if a_now<0 or a_now in done_arms or pull_count[a_now]>=M_i:
                        assigned_arm[mm]=-1
                        pick_new_arm(mm)
                        if assigned_arm[mm]<0:
                            do_takeover(mm)
                        if assigned_arm[mm]==-1: ### not sure about this
                            assigned_arm[mm]=a_now

                reward_vec=np.zeros(self.m)
                regret_vec=np.zeros(self.m)
                noise_vec = math.sqrt(self.variance) * self.rng.normal(size=self.m)
                if self.verbose:
                    self.logger.debug('assigned_arm=%s', assigned_arm)
                # Build proposed arms for this round
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
                            else:
                                if tx_mask[mm] and erasure_bits[mm] == 0:
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

                    reward_val=self.mu[actual_arm] + noise_vec[mm]
                    reward_vec[mm]=reward_val
                    regret_vec[mm]=np.max(self.mu)-self.mu[actual_arm]
                    actual_pull_count[actual_arm] += 1
                    batch_agent_arm_counts[actual_arm][mm] += 1

                    if pull_count[actual_arm]<M_i:
                        pull_count[actual_arm]+=1
                        pull_sums[actual_arm]+=reward_val
                        if pull_count[actual_arm]>=M_i:
                            done_arms.add(actual_arm)
                            done_count+=1

                self.rewards[t,:]=reward_vec
                self.regrets[t,:]=regret_vec
                self.rewards_all[t]=np.sum(reward_vec)
                self.regrets_all[t]=np.sum(regret_vec)
                self.tx_over_time[t] = self.tx_count_total
                self.fb_over_time[t] = self.fb_count_total
                t+=1
                if t>=self.iters:
                    break

            # ============ PHASE 2: chunk finishing with leftover-based takeover, no partial copy ============
            unfinished_arms=[a for a in active_arm_inds if pull_count[a]<M_i]
            if len(unfinished_arms)>0 and len(unfinished_arms)<=self.m and t<self.iters:
                chunk_per_arm=max(1,self.m//(len(unfinished_arms)))
                needed_per_chunk=math.ceil(M_i/float(chunk_per_arm))

                replicate_list=[]
                replicate_count={}
                replicate_sum={}

                for a in unfinished_arms:
                    for cid in range(chunk_per_arm):
                        replicate_list.append((a,cid))
                        replicate_count[(a,cid)]=0
                        replicate_sum[(a,cid)]=0.0
                replicate_active={rc:True for rc in replicate_list}

                assigned_chunk=[(-1,-1)]*self.m
                chunk_taken_over=[False]*self.m

                chunk_delivery_done=[False]*self.m
                chunk_attempts_spent=[0]*self.m
                chunk_num_succ=[0]*self.m

                rep_ptr=0

                def chunk_eta(m_idx):
                    if chunk_taken_over[m_idx]:
                        return 0
                    (rA,rC)=assigned_chunk[m_idx]
                    if rA<0 or not replicate_active.get((rA,rC),False):
                        return 0
                    if replicate_count.get((rA,rC),0)>=needed_per_chunk:
                        return 0
                    if not chunk_delivery_done[m_idx]:
                        return max(0,(self.alphas[m_idx]+(needed_per_chunk - pull_count[rA]))-chunk_attempts_spent[m_idx])
                    else:
                        return max(0,needed_per_chunk-chunk_num_succ[m_idx])

                def find_slowest_chunk_agent(m_idx):
                    slow_idx=-1
                    slow_eta=-1
                    for mm in range(self.m):
                        if mm == m_idx:
                            continue
                        if chunk_taken_over[mm]:
                            continue
                        (ra,rcid)=assigned_chunk[mm]
                        if ra<0 or rcid<0:
                            continue
                        if not replicate_active.get((ra,rcid),False):
                            continue
                        ceta=chunk_eta(mm)
                        if ceta>slow_eta:
                            slow_eta=ceta
                            slow_idx=mm
                    return slow_idx,slow_eta

                def chunk_takeover(m_idx):
                    if chunk_taken_over[m_idx]:
                        return
                    s_idx,s_eta=find_slowest_chunk_agent(m_idx)
                    if s_idx<0 or s_idx==m_idx or s_eta<=0:
                        return
                    my_eta=(self.alphas[m_idx]+(needed_per_chunk - pull_count[assigned_chunk[s_idx][0]]))
                    if my_eta<s_eta:
                        (sA,sC)=assigned_chunk[s_idx]
                        if sA<0 or (sA,sC) not in replicate_active:
                            return
                        chunk_taken_over[s_idx]=True
                        assigned_chunk[m_idx]=(sA,sC)
                        # do not copy partial
                        chunk_delivery_done[m_idx]=False
                        chunk_attempts_spent[m_idx]=0
                        chunk_num_succ[m_idx]=0

                def chunk_done(a,cid):
                    if (a,cid) not in replicate_active:
                        return False
                    if not replicate_active[(a,cid)]:
                        return True
                    if replicate_count[(a,cid)]>=needed_per_chunk:
                        return True
                    if pull_count[a]>=M_i:
                        return True
                    return False

                def remove_all_copies(a):
                    for (xx,cc) in replicate_list:
                        if xx==a:
                            replicate_active[(xx,cc)]=False

                limit_ass=min(self.m,len(replicate_list))
                for iA in range(self.m):
                    if iA < len(replicate_list):
                        assigned_chunk[iA]=replicate_list[iA]
                    else:
                        random_arm = self.rng.choice(unfinished_arms)
                        assigned_chunk[iA] = (random_arm, -1)
                rep_ptr=limit_ass

                def any_active_chunks():
                    for (aa,cc) in replicate_list:
                        if replicate_active.get((aa,cc),False) and replicate_count[(aa,cc)]<needed_per_chunk and pull_count[aa]<M_i:
                            return True
                    return False
                if self.verbose:
                    self.logger.debug('second phase start')
                while any_active_chunks() and t<self.iters:
                    reward_vec=np.zeros(self.m)
                    regret_vec=np.zeros(self.m)

                    for mm in reversed(range(self.m)):
                        if chunk_taken_over[mm]:
                            continue
                        (ra,rcid)=assigned_chunk[mm]
                        if ra<0 or chunk_done(ra,rcid):
                            while rep_ptr<len(replicate_list):
                                cand=replicate_list[rep_ptr]
                                rep_ptr+=1
                                if not chunk_done(*cand):
                                    assigned_chunk[mm]=cand
                                    (ra,rcid)=cand
                                    break
                            else:
                                if self.verbose:
                                    self.logger.debug('entering takeover %s', mm)
                                chunk_takeover(mm)
                                if self.verbose:
                                    self.logger.debug('prev=%s now=%s', ra, assigned_chunk[mm][0])
                            if assigned_chunk[mm][0]<0: ### not sure about this
                                assigned_chunk[mm]=(ra,rcid)

                    if self.verbose:
                        self.logger.debug('assigned_chunk=%s', assigned_chunk)
                    noise_vec = math.sqrt(self.variance) * self.rng.normal(size=self.m)

                    # Build proposed arms for this round of phase 2
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
                        if ra<0:
                            act_arm=self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                        else:
                            if not chunk_delivery_done[mm]:
                                chunk_attempts_spent[mm]+=1
                                budget=(self.alphas[mm]+needed_per_chunk)
                                if chunk_attempts_spent[mm]>budget:
                                    assigned_chunk[mm]=(-1,-1)
                                    act_arm=self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                                else:
                                    if tx_mask[mm] and erasure_bits[mm]==0:
                                        act_arm=ra
                                        self.pulled_ind[mm]=ra
                                        chunk_delivery_done[mm]=True
                                        chunk_num_succ[mm]=1
                                    else:
                                        act_arm=self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]
                            else:
                                if tx_mask[mm] and erasure_bits[mm]==0:
                                    act_arm=ra
                                    self.pulled_ind[mm]=ra
                                    chunk_num_succ[mm]+=1
                                else:
                                    act_arm=self.pulled_ind[mm] if self.pulled_ind[mm] >= 0 else proposed_chunk[mm]

                        reward_val=self.mu[act_arm] + noise_vec[mm]
                        reward_vec[mm]=reward_val
                        regret_vec[mm]=np.max(self.mu)-self.mu[act_arm]
                        actual_pull_count[act_arm]+=1
                        batch_agent_arm_counts[act_arm][mm] += 1

                        if pull_count[act_arm]<M_i:
                            pull_count[act_arm]+=1
                            pull_sums[act_arm]+=reward_val
                            if pull_count[act_arm]>=M_i:
                                remove_all_copies(act_arm)

                        if act_arm==ra and ra>=0 and replicate_active.get((ra,rcid),False):
                            replicate_count[(ra,rcid)]+=1
                            if replicate_count[(ra,rcid)]>=needed_per_chunk or pull_count[ra]>=M_i:
                                replicate_active[(ra,rcid)]=False


                    self.rewards[t,:]=reward_vec
                    self.regrets[t,:]=regret_vec
                    self.rewards_all[t]=np.sum(reward_vec)
                    self.regrets_all[t]=np.sum(regret_vec)
                    self.tx_over_time[t] = self.tx_count_total
                    self.fb_over_time[t] = self.fb_count_total
                    t+=1
                    if t>=self.iters:
                        break

            # end of phase2

            # batch end => merge local counters
            for a in active_arm_inds:
                if pull_count[a]>0:
                    self.arm_sums[a]+=pull_sums[a]
                    self.arm_counts[a]+=pull_count[a]
                    self.k_reward_q[a] = self.arm_sums[a]/float(self.arm_counts[a])

            # partial elimination
            survivors=np.where(self.active_arms==1)[0]
            if len(survivors)==0:
                break
            best_mean=np.max(self.k_reward_q[survivors])
            for a in survivors:
                diff=best_mean-self.k_reward_q[a]
                threshold=C_CONFIDENCE*self.c*math.sqrt(math.log(self.iters*self.m)/(2.0*M_i))
                if diff>threshold:
                    self.active_arms[a]=0

            if self.verbose:
                self.logger.debug('end_time=%s pull_cnt=%s survivors=%s\n%s',
                                  t,
                                  {int(a): int(actual_pull_count[a]) for a in active_arm_inds},
                                  np.where(self.active_arms==1)[0],
                                  batch_agent_arm_counts)

    def reset(self, mu=None, base_actions=None, erasure_seq=None):
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)

        self.k_reward_q = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k, dtype=int)
        self.arm_sums = np.zeros(self.k)
