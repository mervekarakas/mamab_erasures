import numpy as np
import random
import math
import copy

C_CONFIDENCE = 2 #Multiplier in front of the confidence region for SAE

'''
Calculates number of repetitions (delays) for each channel 
based on erasure probabilities (eps) and horizon T (iters)
'''
def calculate_repetitions(eps, iters):

    if type(eps) == float or type(eps) == int:
        eps = [eps]
    eps = np.array(eps)
    alphas = np.ones(len(eps))
    for i in range(len(eps)):
        if eps[i] > 0:
            alphas[i] = np.ceil(2 * np.log(iters) / np.log(1 / eps[i]))
    return alphas.astype(int)


### 
# ALGORITHMS FOR MULTI-AGENT SCENARIO
###
class ucb_ma:
    '''
    UCB algorithm extended to multi-agent setting. no repetition
    '''
    def __init__(self, k, m, iters, var=1, c=1, mu='random', epsilon=0, base=None):
        # Initializations
        self.name = 'MA-UCB'
        self.k = k # Number of arms
        self.m = m # Number of agents
        self.c = c # Exploration parameter
        self.iters = iters # Number of iterations
        self.n = 1 # Iterations count
        self.base = base
        self.k_n = np.ones(k).astype(int) # Sample count for each arm
       
        self.pulled_once = np.zeros(self.k, dtype=int)  #arms pulled for the first time
        
        self.pulled_ind = np.random.randint(k, size=(m,)) if base is None else np.array(base)# Pulled arm indices
        self.pulled_regret = np.zeros(m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(m) # rewards of the current pulled arms
        self.regrets = np.zeros((iters,m)) # Store regrets
        self.rewards = np.zeros((iters,m)) # Store rewards
        self.rewards_all = np.zeros(iters)
        self.regrets_all = np.zeros(iters) # Store sum of regrets from channels
       
        self.k_reward_q = np.zeros(k) # Mean reward for each arm
        self.variance =  var # Noise variance
        self.eps = np.array(epsilon).reshape(-1,) # erasure probability of a given action
       
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.uniform(0, 1, k)
        else:
            print('Problem with mean initialization: ', mu)
            return
       
    def pull(self):
        if self.n <= self.k:
            ct = min(self.m, self.k-self.n+1)
            a_learner = np.arange(self.n-1, self.n - 1 + ct, dtype=int)
            if a_learner.shape[0] < self.m:
               a_learner = np.concatenate((a_learner, np.random.randint(self.k, size=(self.m-ct,))))
             

            is_erasure = np.random.rand(self.m) < self.eps
            arms = copy.deepcopy(a_learner)
            arms[is_erasure] = self.pulled_ind[is_erasure].astype(int)
        else:
            a_learner = np.argmax(self.k_reward_q+self.c*np.sqrt(np.log(1+self.n*(np.log(self.n)**2))/self.k_n))   
            arms = self.pulled_ind.astype(int)
            arms[np.random.rand(self.m) > self.eps] = int(a_learner)
            a_learner = np.full(self.m, a_learner, dtype=int)

        noise = np.sqrt(self.variance)*np.random.normal(0, 1, self.m)
        reward = self.mu[arms] + noise #observed reward in the environment
        self.pulled_reward = reward
        self.pulled_regret = np.max(self.mu)- self.mu[arms]#reward#self.mu[a_learner] #observed regret
        #self.k_n[a_learner] += self.m #learner believes it played a_learner
        self.n += self.m
        self.pulled_ind = arms #storage on the agent side, last action
        #update mean reward of the arm pulled by the learner
        #self.k_reward_q[a_learner] = (self.k_reward_q[a_learner]*(self.k_n[a_learner]-1-self.m)+np.sum(reward))/(self.k_n[a_learner]-1)
        
        unique_arms = np.unique(a_learner)
        for a in unique_arms:
            inds = np.where(a_learner == a)[0]

            self.k_reward_q[a] = (self.k_reward_q[a]*(self.k_n[a]-1) + np.sum(reward[inds])) / (self.k_n[a]- 1 + len(inds))
            self.k_n[a] += len(inds)


    def run(self):
        for i in range(self.iters):
            self.pull()

            self.regrets[i,:] = self.pulled_regret.reshape(1,self.m)
            self.rewards[i,:] = self.pulled_reward.reshape(1,self.m)
            self.regrets_all[i] = np.sum(self.pulled_regret)
            self.rewards_all[i] = np.sum(self.pulled_reward)
  

    def reset(self, mu=None, base_actions=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k).astype(int)
        
        self.k_reward_q = np.zeros(self.k)

        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = np.random.randint(self.k, size=(self.m,)) if self.base is None else np.array(self.base) # Pulled arm indices
        self.pulled_regret = np.zeros(self.m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(self.m) # rewards of the current pulled arms
        self.regrets = np.zeros((self.iters,self.m)) # Store regrets
        self.rewards = np.zeros((self.iters,self.m)) # Store rewards
        self.rewards_all = np.zeros(self.iters)
        self.regrets_all = np.zeros(self.iters) # Store sum of regrets from channels
       

        if type(mu) == str and mu == 'random':
            self.mu = np.random.uniform(0, 1, self.k) #1*np.random.normal(0, 1, self.k)
        elif type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu is not None:
            print('Problem with mean reset value: ', mu)
            return


class LSAE_ma_hor:
    '''
    Successive Arm Elimination with repetition and horizontal assignment. 
    Starting from the first arm, the learner assigns it to all agents and moves onto
    the next arm when required number of effective pulls are received.

    Inputs
    ============================================
    k: number of arms (int)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a uniform distribution with mean = 0.
        Pass a list or array of length = k for user-defined
        values.
    alpha: number of repetitions
    '''
    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0, base=None):
        # Initializations
        self.name = 'MA-LSAE-Horizontal'
        self.k = k # Number of arms
        self.m = m
        self.iters = iters # Number of iterations
        self.c = c
        self.base = base

        self.active_arms = np.ones(k).astype(int) # All arms are active in the beginning

        self.pulled_ind = np.random.randint(k, size=(m,1)) if base is None else np.array(base).reshape(-1,1) # Pulled arm index
        self.pulled_regret = np.zeros(m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(m) # rewards of the current pulled arms
        self.regrets = np.zeros((iters,m)) # Store regrets
        self.rewards = np.zeros((iters,m)) # Store rewards
        self.rewards_all = np.zeros(iters)
        self.regrets_all = np.zeros(iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(k) # Mean reward for each arm
        self.variance =  var # Noise variance
        self.eps = epsilon # erasure probability of a given action
        self.alphas = alphas # repetition parameter

        if type(mu) == list or type(mu).__module__ == np.__name__: # User-defined averages
            self.mu = np.array(mu)
        elif type(mu) == str and mu == 'random': # Draw means from probability distribution
            self.mu = np.random.uniform(0, 1, k)
        else:
            print('Problem with mean initialization: ', mu)
            return


    def run(self):
        counter = 0
        num_pulls = 0
        M_i = 1
        for i in range(self.iters): # T: total number of iterations
          M_i *= 4 
          active_arm_inds = np.where(self.active_arms == 1)[0]

          arms_to_be_assigned = list(active_arm_inds)
          assignments = -1 * np.ones(self.m, dtype=int)
          reps = np.zeros(self.m, dtype=int)

          reward_sums = np.zeros(self.k)
          pull_counts = np.zeros(self.k, dtype=int)

          t = 0
          while (counter < self.iters) and (len(arms_to_be_assigned) > 0):
            t += 1
            for j in range(self.m):
              if assignments[j] == -1:
                assignments[j] = arms_to_be_assigned[0]
              elif pull_counts[assignments[j]] >= M_i:
                if assignments[j] in arms_to_be_assigned:
                  arms_to_be_assigned.remove(assignments[j])
                reps[j] = 0
                if len(arms_to_be_assigned) > 0:
                  assignments[j] = arms_to_be_assigned[0]
              
              a_learner = assignments[j]
              if random.random() < self.eps[j]:
                a = self.pulled_ind[j]
              else:
                a = a_learner

              self.pulled_ind[j] = a
              self.pulled_regret[j] = np.max(self.mu) - self.mu[a]
              self.regrets[counter,j] = self.pulled_regret[j]

              noise = np.sqrt(self.variance)*np.random.normal(0, 1, 1)
              reward = (self.mu[a] + noise)[0]
              if reps[j] < self.alphas[j]-1:
                reps[j] += 1
              else:
                pull_counts[a_learner] += 1
                reward_sums[a_learner] += reward

              self.pulled_reward[j] = reward
              self.rewards[counter,j] = reward

            self.rewards_all[counter] = np.sum(self.rewards[counter,:])
            self.regrets_all[counter] = np.sum(self.regrets[counter,:])
          
            counter += 1
          
          if counter >= self.iters:
             return

          for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a]*num_pulls+reward_sums[a])/(num_pulls+pull_counts[a])
          num_pulls += M_i 
          
          for j in active_arm_inds:
            diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
            if diff > C_CONFIDENCE * self.c * float(np.sqrt(np.log(self.k * self.iters * self.m) / (2 * M_i))):
              self.active_arms[j] = 0
          #print(t, len(active_arm_inds))
         


    def reset(self, mu=None, base_actions=None):
        # Resets results while keeping settings
        self.active_arms = np.ones(self.k)

        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = np.random.randint(self.k, size=(self.m,1)) if self.base is None else np.array(self.base) # Pulled arm indices
        self.pulled_regret = np.zeros(self.m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(self.m) # rewards of the current pulled arms
        self.regrets = np.zeros((self.iters,self.m)) # Store regrets
        self.rewards = np.zeros((self.iters,self.m)) # Store rewards
        self.rewards_all = np.zeros(self.iters)
        self.regrets_all = np.zeros(self.iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(self.k) # Mean reward for each arm
        
        if type(mu) == str and mu == 'random':
            self.mu = np.random.uniform(0, 1, self.k) #1*np.random.normal(0, 1, self.k)
        elif type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu is not None:
            print('Problem with mean reset value: ', mu)
            return


class LSAE_ma_ver:
    '''
    Successive Arm Elimination for multi-agent setting with vertical assignment. 
    This method assigns all pulls of the next arm to the first available agent, 
    if there are more than one avaialable agent, it chooses the one with smaller delay. 
    If no arm left to assign, it plays random arms on the remaning agents until assigned 
    arm pulls are complete.

    Inputs
    ============================================
    k: number of arms (int)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a uniform distribution with mean = 0.
        Pass a list or array of length = k for user-defined
        values.
    alphas: number of repetitions for each channel
    '''
    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0, base=None):
        # Initializations
        self.name = 'MA-LSAE-Vertical'
        self.k = k 
        self.m = m
        self.iters = iters 
        self.c = c
        self.base = base

        self.active_arms = np.ones(k).astype(int) # All arms are active in the beginning

        self.pulled_ind = np.random.randint(k, size=(m,1)) if base is None else np.array(base).reshape(-1,1) # Pulled arm index
        self.pulled_regret = np.zeros(m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(m) # rewards of the current pulled arms
        self.regrets = np.zeros((iters,m)) # Store regrets
        self.rewards = np.zeros((iters,m)) # Store rewards
        self.rewards_all = np.zeros(iters)
        self.regrets_all = np.zeros(iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(k) # Mean reward for each arm
        self.variance =  var # Noise variance
        self.eps = epsilon # erasure probability of a given action
        self.alphas = alphas # repetition parameter

        if type(mu) == list or type(mu).__module__ == np.__name__: # User-defined averages
            self.mu = np.array(mu)
        elif type(mu) == str and mu == 'random': # Draw means from probability distribution
            self.mu = np.random.uniform(0, 1, k)
        else:
            print('Problem with mean initialization: ', mu)
            return


    def run(self):
        counter = 0
        num_pulls = 0
        M_i = 1
        for i in range(self.iters): # T: total number of iterations
          M_i *= 4 
          active_arm_inds = np.where(self.active_arms == 1)[0]

          arms_to_be_assigned = list(active_arm_inds)
          assignments = -1 * np.ones(self.m, dtype=int)
          reps = np.zeros(self.m, dtype=int)

          reward_sums = np.zeros(self.k)
          pull_counts = np.zeros(self.k, dtype=int)

          t = 0
          while (counter < self.iters) and np.any(pull_counts[active_arm_inds] < M_i):
            t += 1
            for j in range(self.m):
              if assignments[j] == -1:
                if len(arms_to_be_assigned) > 0:
                  assignments[j] = arms_to_be_assigned[0]
                  arms_to_be_assigned.remove(assignments[j])
                else:
                  assignments[j] = self.k #indicates dummy assignment, this agent will play random arms
              elif assignments[j] == self.k:
                reps[j] = 0
              elif pull_counts[assignments[j]] >= M_i:      
                reps[j] = 0
                if len(arms_to_be_assigned) > 0:
                  assignments[j] = arms_to_be_assigned[0]
                  arms_to_be_assigned.remove(assignments[j])
        
              
              a_learner = assignments[j] if assignments[j] < self.k else random.choice(active_arm_inds)
              if random.random() < self.eps[j]:
                a = self.pulled_ind[j]
              else:
                a = a_learner

              self.pulled_ind[j] = a
              self.pulled_regret[j] = np.max(self.mu) - self.mu[a]
              self.regrets[counter,j] = self.pulled_regret[j]

              
              noise = np.sqrt(self.variance)*np.random.normal(0, 1, 1)
              reward = (self.mu[a] + noise)[0]
              if reps[j] < self.alphas[j]-1:
                reps[j] += 1
              else:
                pull_counts[a_learner] += 1
                reward_sums[a_learner] += reward

              self.pulled_reward[j] = reward
              self.rewards[counter,j] = reward

            self.rewards_all[counter] = np.sum(self.rewards[counter,:])
            self.regrets_all[counter] = np.sum(self.regrets[counter,:])
          
            counter += 1
          
          if counter >= self.iters:
             return

          for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a]*num_pulls+reward_sums[a])/(num_pulls+pull_counts[a])
          num_pulls += M_i 
          
          for j in active_arm_inds:
            diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
            if diff > C_CONFIDENCE * self.c *  np.sqrt(np.log(self.k * self.iters * self.m) / (2 * M_i)):
              self.active_arms[j] = 0
          



    def reset(self, mu=None, base_actions=None):
        # Resets results while keeping settings
        self.active_arms = np.ones(self.k)

        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = np.random.randint(self.k, size=(self.m,1)) if self.base is None else np.array(self.base).reshape(-1,1) # Pulled arm indices
        self.pulled_regret = np.zeros(self.m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(self.m) # rewards of the current pulled arms
        self.regrets = np.zeros((self.iters,self.m)) # Store regrets
        self.rewards = np.zeros((self.iters,self.m)) # Store rewards
        self.rewards_all = np.zeros(self.iters)
        self.regrets_all = np.zeros(self.iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(self.k) # Mean reward for each arm
        
        if type(mu) == str and mu == 'random':
            self.mu = np.random.uniform(0, 1, self.k) #1*np.random.normal(0, 1, self.k)
        elif type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu is not None:
            print('Problem with mean reset value: ', mu)
            return


class BatchSP2:
    '''
    BatchSP2 method provided in our paper.

    Inputs
    ============================================
    k: number of arms (int)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a uniform distribution with mean = 0.
        Set to "sequence" for the means to be ordered from
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    alphas: number of repetitions
    epsilon: erasure probabilities
    '''
    def __init__(self, k, m, iters, alphas, var=1, c=1, mu='random', epsilon=0, base=None):
        # Initializations
        self.name = 'MA-LSAE-Scheduled'
        self.k = k # Number of arms
        self.m = m
        self.iters = iters # Number of iterations
        self.c = c
        self.base = base

        self.active_arms = np.ones(k).astype(int) # All arms are active in the beginning

        self.pulled_ind = np.random.randint(k, size=(m,)) if base is None else np.array(base) # Pulled arm index
        self.pulled_regret = np.zeros(m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(m) # rewards of the current pulled arms
        self.regrets = np.zeros((iters,m)) # Store regrets
        self.rewards = np.zeros((iters,m)) # Store rewards
        self.rewards_all = np.zeros(iters)
        self.regrets_all = np.zeros(iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(k) # Mean reward for each arm
        self.variance =  var # Noise variance
        self.eps = epsilon # erasure probability of a given action
        self.alphas = alphas # repetition parameter

        if type(mu) == list or type(mu).__module__ == np.__name__: # User-defined averages
            self.mu = np.array(mu)
        elif type(mu) == str and mu == 'random': # Draw means from probability distribution
            self.mu = np.random.uniform(0, 1, k)
        else:
            print('Problem with mean initialization: ', mu)
            return

    def schedule(self, M_i):
        # recognize arm to be assigned and time to schedule full arm pulls
        active_arm_inds = np.where(self.active_arms == 1)[0]
        random.shuffle(active_arm_inds)
        T_i = math.ceil(M_i * len(active_arm_inds) / np.sum(M_i / (self.alphas - 1 + M_i)))
        assigned = np.zeros((self.m, self.k), dtype=int) #how many pulls are assigned to which agent
        req_pulls = np.zeros(self.k, dtype=int)
        req_pulls[self.active_arms == 1] = M_i #only active arms need to be scheduled
        eff_pulls = np.zeros(self.k, dtype=int) #current efffective assignments of eac arm

        ind = 0 #start with the first active arm
        #for each agent, assign all pulls of the next active arm until time limit reached
        for i in range(self.m):
            t_end = 0
            p = self.alphas[i]-1 + M_i
            while t_end + p <= T_i:
                arm_idx = active_arm_inds[ind]
                assigned[i,arm_idx] = p
                eff_pulls[arm_idx] = M_i
                t_end += p
                ind += 1
        k_hat = len(active_arm_inds) - ind #remaining unassigned arms
        if k_hat > 0:
          chunk_per_arm = max(1, math.floor(self.m / (2 * k_hat)))
          chunk_size = math.ceil(M_i / chunk_per_arm)
          agents_half = max(1, math.floor(self.m / 2))

          chunk_per_agent = math.ceil(k_hat * chunk_per_arm / agents_half)

          agent_idx = 0
          agent_chk = 0

          for i in range(ind, len(active_arm_inds)):
            arm_idx = active_arm_inds[i]
            while eff_pulls[arm_idx] < req_pulls[arm_idx]:
              num_chks = min(chunk_per_agent-agent_chk, chunk_per_arm)
              num_pulls = min(M_i - eff_pulls[arm_idx] , num_chks * chunk_size)

              assigned[agent_idx, arm_idx] = self.alphas[agent_idx] - 1 + num_pulls
              eff_pulls[arm_idx] += num_pulls
              agent_chk += num_chks

              if agent_chk >= chunk_per_agent:
                agent_idx += 1
                agent_chk = 0

        #print(assigned)
        max_iter = max(np.sum(assigned, axis=1))
        schedule = -1 * np.ones((max_iter, self.m)).astype(int)
  
        for i in range(self.m):
            ctr_ = 0
            for j in range(self.k):
                if assigned[i][j] > 0:
                    schedule[ctr_:ctr_ + assigned[i][j], i] = j
                    ctr_ += assigned[i][j]
            if ctr_ < max_iter:
                schedule[ctr_:, i] = np.random.choice(np.where(self.active_arms == 1)[0], max_iter-ctr_)
        
        return schedule

    def run(self):
        counter = 0
        num_pulls = 0
        #M_i = 1
        M_i = int(np.ceil(2 * np.log(self.iters * self.m )) / 4)
            
        for i in range(self.iters): # T: total number of iterations
          M_i *= 4 
          active_arm_inds = np.where(self.active_arms == 1)[0]

          #keep track of effective number of pulls and rewards for the current batch
          reward_sums = np.zeros(self.k)
          pull_counts = np.zeros(self.k, dtype=int)

          arr_of_schd = self.schedule(M_i)
          reps = np.zeros(self.m, dtype=int)
          
          for t in range(arr_of_schd.shape[0]):
            if counter >= self.iters:
              return
            
            arms = copy.deepcopy(arr_of_schd[t,:])
            indcs = np.random.rand(self.m) < self.eps #erasures
            arms[indcs] = self.pulled_ind[indcs].astype(int)
            
            #noise = np.zeros(self.m)
            noise = np.sqrt(self.variance)*np.random.normal(0, 1, self.m)
            reward = self.mu[arms] + noise #observed reward in the environment
            
            self.pulled_reward = reward
            self.pulled_regret = np.max(self.mu)- self.mu[arms]#reward#self.mu[a_learner] #observed regret
            self.pulled_ind = arms #storage on the agent side, last action

            self.regrets[counter,:] = self.pulled_regret
            self.regrets_all[counter] = np.sum(self.pulled_regret)

            self.rewards[counter,:] = reward
            self.rewards_all[counter] = np.sum(reward)

            for j in range(self.m):
              a_learner = arr_of_schd[t,j]

              if reps[j] < self.alphas[j]-1: #wait until repetitions are complete
                reps[j] += 1
              elif pull_counts[a_learner] < M_i: #collect rewards after reps are completed
                reward_sums[a_learner] += reward[j]
                pull_counts[a_learner] += 1

              #if there is an arm change, zero the repetition
              if t+1 < arr_of_schd.shape[0] and arr_of_schd[t+1,j] != arr_of_schd[t,j]:
                reps[j] = 0
                
            counter += 1
          
          if counter >= self.iters:
             return

          #print('pull_counts',pull_counts)
          for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a]*num_pulls+reward_sums[a])/(num_pulls+pull_counts[a])
          num_pulls += M_i 
          
          for j in active_arm_inds:
            diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
          
            if diff > C_CONFIDENCE * self.c * float(np.sqrt(np.log(self.iters * self.m) / (2 * M_i))):
              self.active_arms[j] = 0
        #  print(t, len(active_arm_inds), num_pulls)
          

    def reset(self, mu=None, base_actions=None):
        # Resets results while keeping settings
        self.active_arms = np.ones(self.k)

        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = np.random.randint(self.k, size=(self.m,)) if self.base is None else np.array(self.base) # Pulled arm indices
        self.pulled_regret = np.zeros(self.m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(self.m) # rewards of the current pulled arms
        self.regrets = np.zeros((self.iters,self.m)) # Store regrets
        self.rewards = np.zeros((self.iters,self.m)) # Store rewards
        self.rewards_all = np.zeros(self.iters)
        self.regrets_all = np.zeros(self.iters) # Store sum of regrets from channels
        
        self.k_reward_q = np.zeros(self.k) # Mean reward for each arm
        
        if type(mu) == str and mu == 'random':
            self.mu = np.random.uniform(0, 1, self.k) #1*np.random.normal(0, 1, self.k)
        elif type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu is not None:
            print('Problem with mean reset value: ', mu)
            return


class Vanilla_SAE_ma:
    '''
    Successive Arm Elimination without repetition for multi-agent setting.
    
    For each batch, calculates the total number of arms and assigns it equally to all agents. 
    Placement of arm pulls start from the first agent and continues, e.g., assume each agent will 
    pull N times, min(N, M_i) of the first arm is assigned to the first agent where M_i is the 
    number of pulls in batch i.

    Inputs
    ============================================
    k: number of arms (int)
    m: number of agents (int)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a uniform distribution with mean = 0.
        Pass a list or array of length = k for user-defined
        values.
    epsilon: erasure probabilities of channels
    '''
    def __init__(self, k, m, iters, var=1, c=1, mu='random', epsilon=0, base=None):
        # Initializations
        self.name = 'MA-SAE'
        self.k = k 
        self.m = m
        self.iters = iters 
        self.c = c
        self.base = base

        self.active_arms = np.ones(k).astype(int) # All arms are active in the beginning

        self.pulled_ind = np.random.randint(k, size=(m,)) if base is None else np.array(base) # Pulled arm index
        self.pulled_regret = np.zeros(m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(m) # rewards of the current pulled arms
        self.regrets = np.zeros((iters,m)) # Store regrets
        self.rewards = np.zeros((iters,m)) # Store rewards
        self.rewards_all = np.zeros(iters)
        self.regrets_all = np.zeros(iters) # Store sum of regrets from channels
       
        self.k_reward_q = np.zeros(k) # Mean reward for each arm
        self.variance =  var # Noise variance
        self.eps = epsilon # erasure probability of a given action
       

        if type(mu) == list or type(mu).__module__ == np.__name__: # User-defined averages
            self.mu = np.array(mu)
        elif type(mu) == str and mu == 'random': # Draw means from probability distribution
            self.mu = np.random.uniform(0, 1, k)
        else:
            print('Problem with mean initialization: ', mu)
            return


    def pull(self, n_m):
        num_pulls_per_agent = np.ceil(np.sum(self.active_arms) * n_m / self.m)
        assigned = np.zeros((self.m, self.k), dtype=int)
        req_pulls = np.zeros(self.k, dtype=int)
        req_pulls[self.active_arms == 1] = n_m
        ind = 0
        for i in range(self.m):
            c = 0
            while c < num_pulls_per_agent and ind < self.k:
                num_pull = min(num_pulls_per_agent-c, req_pulls[ind])
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
                arr_of_schd[ctr_:, i] = np.random.choice(np.where(self.active_arms == 1)[0], max_iter-ctr_)
        
        return arr_of_schd     

    def run(self):
        counter = 0
        num_pulls = 0
        del_i = 2
        for i in range(self.iters): # T: total number of iterations
            del_i /= 2
            active_arm_inds = np.where(self.active_arms == 1)[0]
            n_m = self.c * int(np.ceil(2 * np.log(self.iters * self.m * (del_i ** 2) ) / (del_i ** 2)))
            
            reward_sums = np.zeros(self.k)
            reward_cts = np.zeros(self.k).astype(int)

            arr_of_schd = self.pull(n_m)
            
            for t in range(arr_of_schd.shape[0]):
                if counter >= self.iters:
                    return
                
                arms = arr_of_schd[t,:]
                indcs = np.random.rand(self.m) < self.eps
                arms[indcs] = self.pulled_ind[indcs].astype(int)


                noise = np.sqrt(self.variance)*np.random.normal(0, 1, self.m)
                reward = self.mu[arms] + noise #observed reward in the environment
                
                self.pulled_reward = reward
                self.pulled_regret = np.max(self.mu)- self.mu[arms]#reward#self.mu[a_learner] #observed regret
                self.pulled_ind = arms #storage on the agent side, last action

                self.regrets[counter,:] = self.pulled_regret
                self.regrets_all[counter] = np.sum(self.pulled_regret)

                self.rewards[counter,:] = reward
                self.rewards_all[counter] = np.sum(reward)

                for j in range(self.m):
                    a = arr_of_schd[t,j]
                    if reward_cts[a] < n_m:
                        reward_sums[a] += reward[j]
                        reward_cts[a] += 1
                
                counter += 1
                
            #update mean reward of the arm pulled by the learner
            for a in active_arm_inds:
                self.k_reward_q[a] = (self.k_reward_q[a]*num_pulls+reward_sums[a])/(num_pulls+n_m)
            num_pulls += n_m

        
            for j in active_arm_inds:
                diff = np.max(self.k_reward_q[active_arm_inds]) - self.k_reward_q[j]
                if diff > np.sqrt(2 * np.log((del_i ** 2) * self.iters * self.m) / n_m):
                    self.active_arms[j] = 0


    def reset(self, mu=None, base_actions=None):
        # Resets results while keeping settings
        self.k_reward_q = np.zeros(self.k)
        self.active_arms = np.ones(self.k).astype(int)

        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = np.random.randint(self.k, size=(self.m,)) if self.base is None else np.array(self.base) # Pulled arm indices
        self.pulled_regret = np.zeros(self.m) # Regret of current pulled arms
        self.pulled_reward = np.zeros(self.m) # rewards of the current pulled arms
        self.regrets = np.zeros((self.iters,self.m)) # Store regrets
        self.rewards = np.zeros((self.iters,self.m)) # Store rewards
        self.rewards_all = np.zeros(self.iters)
        self.regrets_all = np.zeros(self.iters) # Store sum of regrets from channels
       
        if type(mu) == str and mu == 'random':
            self.mu = np.random.uniform(0, 1, self.k) #1*np.random.normal(0, 1, self.k)
        elif type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu is not None:
            print('Problem with mean reset value: ', mu)
            return
