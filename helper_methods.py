from mab_algorithms import *
import pickle
import os
import matplotlib.pyplot as plt
import copy

EXP_FOLDER = './Experiments/'
CONFIG = {
   
}

stored_means_file = 'means_stored.pickle'
means = pickle.load(open(stored_means_file,'rb'))


'''
alg_name: which algorithm to use; 'UCB' or 'SAE' (str)
rep: with repetition or not (bool). If False for SAE, vanilla SAE is run.
mode: {'vertical', 'horizontal', 'scheduled'} for SAE; not used for UCB
'''
def init_bandit(alg_name, rep, mode, iters, k, m, mu, eps, var, delta, c, base_actions):
    alphas = calculate_repetitions(eps, iters) 
    if alg_name.upper() == 'UCB':
            return ucb_ma(k, m, iters, var=var, c=c, mu=mu, epsilon=eps)          
    elif alg_name.upper() == 'SAE':
        if not rep: # vanilla SAE
            return Vanilla_SAE_ma(k, m, iters=iters, var=var, mu=mu, epsilon=eps, c=c, base=base_actions)
        elif mode.lower() == 'vertical':
            return LSAE_ma_ver(k, m, iters, alphas,var=var, mu=mu, epsilon=eps, c=c, base=base_actions)
        elif mode.lower() == 'horizontal':
            return LSAE_ma_hor(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c, base=base_actions)
        elif mode.lower() == 'scheduled':
            return BatchSP2(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c, base=base_actions)
        else:
            print('Error in the mode for SAE with repetition, mode:', mode)
            return  
    else:
        print('Unknown algorithm name:', alg_name)
        return None


def run_episodes(alg_name, rep, mode, iters, k, episodes, m=1, mu='random', eps=0, var=1, delta=0.01, c=1, folder_name=None, base_actions=None):
    avg_regret = np.zeros((iters,m))
    regrets = []
    arm_means_lst = []

    bandit = init_bandit(alg_name, rep, mode, iters, k, m, mu, eps, var, delta, c, base_actions)

    f_name = bandit.name if folder_name is None else folder_name + '/' + bandit.name

    for i in range(episodes):
        bandit.reset(mu)
        bandit.run()

        arm_means_lst.append(bandit.mu)
        regrets.append(copy.deepcopy(bandit.regrets))
        avg_regret += bandit.regrets / episodes

    vars = {
        'name' : bandit.name,
        'k' : bandit.k,
        'm' : bandit.m,
        'iters' : bandit.iters,
        'c' : bandit.c,
        'var': bandit.variance,
        'epsilon' : bandit.eps,

        'episodes' : episodes,
        'mu_lst' : arm_means_lst,
        'regret' : avg_regret,
        'episode_regrets' : regrets
    }

    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)
    if not os.path.exists(EXP_FOLDER+f_name):
        os.makedirs(EXP_FOLDER+f_name)

    file_loc = EXP_FOLDER + f_name + ".pickle"
    out_file = open(file_loc,"wb")
    pickle.dump(vars, out_file)
    out_file.close()

    print('Experiment ended:', k, m, iters, eps)
    return bandit.name, file_loc

'''
Run the algorithm with the stored means (for consistency across algorithms)
Change file name at the beginning of this .py file to run with other stored means
'''
def run_predefined_means(alg_name, rep, mode, iters, k, episodes, m=1, mus=None, eps=0, var=1, delta=0.01, c=1, folder_name=None, base_actions=None):
    avg_regret = np.zeros((iters,m))
    regrets = []
    arm_means_lst = []

    if mus is None:
        mus = means

    bandit = init_bandit(alg_name, rep, mode, iters, k, m, 'random', eps, var, delta, c, base_actions)

    f_name = bandit.name if folder_name is None else folder_name + '/' + bandit.name

    for i in range(episodes):
        bandit.reset(mus[str(k)][i%len(mus)])
        bandit.run()

        arm_means_lst.append(bandit.mu)
        regrets.append(copy.deepcopy(bandit.regrets))
        avg_regret += bandit.regrets / episodes

    vars = {
        'name' : bandit.name,
        'k' : bandit.k,
        'm' : bandit.m,
        'iters' : bandit.iters,
        'c' : bandit.c,
        'var': bandit.variance,
        'epsilon' : bandit.eps,

        'episodes' : episodes,
        'mu_lst' : arm_means_lst,
        'regret' : avg_regret,
        'episode_regrets' : regrets
    }

   
    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)
    if not os.path.exists(EXP_FOLDER+f_name):
        os.makedirs(EXP_FOLDER+f_name)

    file_loc = EXP_FOLDER + f_name + ".pickle"
    out_file = open(file_loc,"wb")
    pickle.dump(vars, out_file)
    out_file.close()

    print('Experiment ended:', k, m, iters, eps)
    return bandit.name, file_loc


def plot_and_save(results, labels, title='Regret vs Horizon', info=None, colors=None, linestyles=None, log_scale=False, f_name='results.png'):
    clrs = colors if colors is not None else ["black", "blue", "red", "green", "purple", "orange", "olive", "darkmagenta", "mediumpurple"]
    lstyles = linestyles if linestyles is not None else ["solid", ":", "--", "-.", (0, (3, 1, 1, 1, 1)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1))]
    
    plt.figure()
    clr_idx = 0
    for i in range(len(results)):
        if log_scale:
            plt.semilogy(range(1,results[i].shape[0]+1),np.cumsum(results[i]), linewidth=2, markevery=5000, color=clrs[clr_idx], label=labels[i])
        else: 
            plt.plot(range(1,results[i].shape[0]+1),np.cumsum(results[i]), linewidth=2, markevery=5000, color=clrs[clr_idx], label=labels[i])
        clr_idx = (clr_idx + 1) % len(clrs)

    plt.xlabel("Iterations")
    plt.ylabel("Regret R_T")
    title = title if info is None else title + info
    plt.title(title)
    plt.legend()

    plt.savefig(f_name)

    return f_name
