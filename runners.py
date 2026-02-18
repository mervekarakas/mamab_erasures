"""
runners — Episode execution for multi-agent bandit experiments.
"""

import os
import copy
import random
import logging

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from models import FEEDBACK_NONE, FEEDBACK_BEACON
from models.factory import init_bandit
from utils import generate_erasure_sequence_multi

logger = logging.getLogger(__name__)

EXP_FOLDER = './Experiments/'


def _run_episode_worker(args):
    """
    Helper for process-based episode evaluation.
    """
    (algs, iters, k, m, var, delta, c, base_actions, feedback_modes, mu_val, eps_val, sequence, rng_seed_ep) = args
    ep_results = {}
    list_modes = isinstance(feedback_modes, (list, tuple))
    for idx, alg in enumerate(algs):
        if len(alg) == 4:
            alg_name, rep, mode, fb_override = alg
        else:
            alg_name, rep, mode = alg
            fb_override = None
        mode_l = mode.lower()
        fb_default = feedback_modes[idx] if list_modes else feedback_modes
        if fb_override is not None:
            fb_mode = fb_override
        elif mode_l == 'scheduled':
            fb_mode = FEEDBACK_NONE
        else:
            fb_mode = fb_default

        if mode_l in {'tpg', 'tpg-new', 'tpg-old', 'tpg-simplified', 'tpg-simple'} and fb_mode == FEEDBACK_NONE:
            fb_mode = FEEDBACK_BEACON

        rng_for_alg = None
        if rng_seed_ep is not None:
            rng_for_alg = np.random.default_rng(rng_seed_ep)

        bandit = init_bandit(alg_name, rep, mode, iters, k, m, mu_val, eps_val, var, delta, c, base_actions, fb_mode, rng=rng_for_alg)
        bandit.reset(mu=mu_val, base_actions=base_actions, erasure_seq=sequence)
        bandit.run()
        ep_results[alg] = {
            "regret": bandit.regrets.reshape(-1, m),
            "mu": bandit.mu,
            "tx": getattr(bandit, "tx_over_time", None),
            "fb": getattr(bandit, "fb_over_time", None),
            "name": bandit.name,
        }
    return ep_results


def run_episodes(alg_name, rep, mode, iters, k, episodes, m=1, mu='random', eps=0, var=1, delta=0.01, c=1, sequences=None, folder_name=None, base_actions=None, feedback_mode=FEEDBACK_NONE, rng_seed=None):
    """
    Runs episodes for a single algorithm; supports optional process-based parallelism.
    """
    avg_regret = np.zeros((iters, m))
    regrets = []
    arm_means_lst = []

    avg_tx = np.zeros(iters)
    avg_fb = np.zeros(iters)

    random_means = (mu == "random")

    mu_list = []
    seq_list = []
    rng_list = None
    if rng_seed is not None:
        base_rng = np.random.default_rng(rng_seed)
        rng_list = base_rng.integers(0, 2**32, size=episodes, dtype=np.uint32)

    for ep_idx in range(episodes):
        if sequences:
            seq = sequences[ep_idx]
        else:
            seq = generate_erasure_sequence_multi(iters, m, eps)
        seq_list.append(seq)

        if random_means:
            gap_target = 10 * (k ** 0.5) / (iters ** 0.5)
            if gap_target >= 1:
                gap_target = 0.9
            candidate = np.random.uniform(0, 1, k)
            sorted_indices = np.argsort(candidate)[-2:]
            max1_idx, max2_idx = sorted_indices[::-1]
            retries = 0
            while candidate[max1_idx] - candidate[max2_idx] < gap_target and retries < 10000:
                candidate = np.random.uniform(0, 1, k)
                sorted_indices = np.argsort(candidate)[-2:]
                max1_idx, max2_idx = sorted_indices[::-1]
                retries += 1
            mu_val = candidate.tolist()
            random.shuffle(mu_val)
        else:
            mu_val = mu
        mu_list.append(mu_val)

    def _run_single_episode(ep_idx):
        mode_l = mode.lower()
        fb_mode = feedback_mode
        if mode_l in {'tpg', 'tpg-new', 'tpg-old', 'tpg-simplified', 'tpg-simple'} and fb_mode == FEEDBACK_NONE:
            fb_mode = FEEDBACK_BEACON
        rng_arg = None
        if rng_list is not None:
            rng_arg = np.random.default_rng(rng_list[ep_idx])
        bandit = init_bandit(alg_name, rep, mode, iters, k, m, mu_list[ep_idx], eps, var, delta, c, base_actions, fb_mode, rng=rng_arg)
        if seq_list[ep_idx] is not None:
            bandit.reset(mu=mu_list[ep_idx], base_actions=base_actions, erasure_seq=seq_list[ep_idx])
        else:
            bandit.reset(mu=mu_list[ep_idx], base_actions=base_actions)
        bandit.run()
        return {
            "regret": bandit.regrets.reshape(-1, m),
            "mu": bandit.mu,
            "tx": getattr(bandit, "tx_over_time", None),
            "fb": getattr(bandit, "fb_over_time", None),
            "name": bandit.name,
        }

    use_parallel = os.environ.get("RUN_MAB_PARALLEL", "").lower() in ("1", "true", "yes")
    if use_parallel and episodes > 1:
        max_workers = os.cpu_count() or 2
        max_env = os.environ.get("RUN_MAB_MAX_WORKERS")
        if max_env:
            try:
                max_workers = min(max_workers, max(1, int(max_env)))
            except ValueError:
                pass
        workers = min(max_workers, episodes)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for data in pool.map(_run_single_episode, range(episodes)):
                avg_regret += data["regret"] / episodes
                regrets.append(copy.deepcopy(data["regret"]))
                if data["tx"] is not None:
                    avg_tx += data["tx"] / episodes
                if data["fb"] is not None:
                    avg_fb += data["fb"] / episodes
                arm_means_lst.append(data["mu"])
                bandit_name = data["name"]
    else:
        bandit_name = None
        for ep_idx in range(episodes):
            data = _run_single_episode(ep_idx)
            avg_regret += data["regret"] / episodes
            regrets.append(copy.deepcopy(data["regret"]))
            if data["tx"] is not None:
                avg_tx += data["tx"] / episodes
            if data["fb"] is not None:
                avg_fb += data["fb"] / episodes
            arm_means_lst.append(data["mu"])
            bandit_name = data["name"] if bandit_name is None else bandit_name

    if bandit_name is None:
        bandit_name = f"{alg_name}_{mode}"

    f_name = bandit_name if folder_name is None else folder_name + '/' + bandit_name + '_' + feedback_mode

    vars = {
        'name': bandit_name,
        'k': k,
        'm': m,
        'iters': iters,
        'c': c,
        'var': var,
        'epsilon': eps,
        'episodes': episodes,
        'mu_lst': arm_means_lst,
        'regret': avg_regret,
        'episode_regrets': regrets,
        'avg_tx': avg_tx,
        'avg_fb': avg_fb,
    }

    import pickle
    os.makedirs(EXP_FOLDER, exist_ok=True)
    os.makedirs(EXP_FOLDER + f_name, exist_ok=True)

    file_loc = EXP_FOLDER + f_name + ".pickle"
    with open(file_loc, "wb") as out_file:
        pickle.dump(vars, out_file)

    logger.info('Experiment ended: k=%s m=%s iters=%s eps=%s', k, m, iters, eps)
    return bandit_name, file_loc


def run_episodes_with_same_erasures(algs, iters, k, episodes, m=1, mu='random', eps=0, var=1, delta=0.01, c=1, folder_name=None, base_actions=None, mus=None, feedback_mode=FEEDBACK_BEACON, rng_seed=None):
    """
    Run episodes while keeping identical scheduling/feedback logic across algorithms.
    """
    avg_regret = {alg: np.zeros((iters, m)) for alg in algs}
    regrets = {alg: [] for alg in algs}
    arm_means_lst = {alg: [] for alg in algs}
    bandit_names = {}

    random_means = (mu == "random")
    random_eps = (isinstance(eps, str) and eps == 'random')

    avg_tx = {alg: np.zeros(iters) for alg in algs}
    avg_fb = {alg: np.zeros(iters) for alg in algs}

    eps_list = []
    mu_list = []
    seq_list = []
    for _ in range(episodes):
        eps_val = np.sort(np.random.random(m)) if random_eps else eps
        eps_list.append(eps_val)
        sequence = generate_erasure_sequence_multi(iters, m, eps_val)
        seq_list.append(sequence)

        if mus:
            mu_val = mus[len(mu_list)]
        elif random_means:
            gap_target = 10 * (k ** 0.5) / (iters ** 0.5)
            if gap_target >= 1:
                gap_target = 0.9
            candidate = np.random.uniform(0, 1, k)
            sorted_indices = np.argsort(candidate)[-2:]
            max1_idx, max2_idx = sorted_indices[::-1]
            retries = 0
            while candidate[max1_idx] - candidate[max2_idx] < gap_target and retries < 10000:
                candidate = np.random.uniform(0, 1, k)
                sorted_indices = np.argsort(candidate)[-2:]
                max1_idx, max2_idx = sorted_indices[::-1]
                retries += 1
            mu_val = candidate.tolist()
            random.shuffle(mu_val)
            logger.debug("Random mu drawn: %s", mu_val)
        else:
            mu_val = mu
        mu_list.append(mu_val)

    fb_modes = feedback_mode
    if isinstance(feedback_mode, (list, tuple, np.ndarray)):
        if len(feedback_mode) != len(algs):
            raise ValueError("feedback_mode list/tuple length must match algs length")
        fb_modes = list(feedback_mode)

    rng_seeds = None
    if rng_seed is not None:
        base_rng = np.random.default_rng(rng_seed)
        rng_seeds = base_rng.integers(0, 2**32, size=episodes, dtype=np.uint32)

    use_parallel = os.environ.get("RUN_MAB_PARALLEL", "").lower() in ("1", "true", "yes")
    if use_parallel and episodes > 1:
        max_workers = os.cpu_count() or 2
        max_env = os.environ.get("RUN_MAB_MAX_WORKERS")
        if max_env:
            try:
                max_workers = min(max_workers, max(1, int(max_env)))
            except ValueError:
                pass
        workers = min(max_workers, episodes)
        tasks = (
            (algs, iters, k, m, var, delta, c, base_actions, fb_modes, mu_list[i], eps_list[i], seq_list[i], None if rng_seeds is None else int(rng_seeds[i]))
            for i in range(episodes)
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for ep_results in pool.map(_run_episode_worker, tasks):
                for alg, data in ep_results.items():
                    avg_regret[alg] += data["regret"] / episodes
                    regrets[alg].append(copy.deepcopy(data["regret"]))
                    if data["tx"] is not None:
                        avg_tx[alg] += data["tx"] / episodes
                    if data["fb"] is not None:
                        avg_fb[alg] += data["fb"] / episodes
                    arm_means_lst[alg].append(data["mu"])
                    bandit_names.setdefault(alg, data["name"])
    else:
        for ep_idx in range(episodes):
            ep_results = _run_episode_worker(
                (algs, iters, k, m, var, delta, c, base_actions, fb_modes, mu_list[ep_idx], eps_list[ep_idx], seq_list[ep_idx], None if rng_seeds is None else int(rng_seeds[ep_idx]))
            )
            for alg, data in ep_results.items():
                avg_regret[alg] += data["regret"] / episodes
                regrets[alg].append(copy.deepcopy(data["regret"]))
                if data["tx"] is not None:
                    avg_tx[alg] += data["tx"] / episodes
                if data["fb"] is not None:
                    avg_fb[alg] += data["fb"] / episodes
                arm_means_lst[alg].append(data["mu"])
                bandit_names.setdefault(alg, data["name"])

    vars = {
        alg: {
            'regret': avg_regret[alg],
            'episode_regrets': regrets[alg],
            'name': bandit_names[alg],
            'avg_tx': avg_tx[alg],
            'avg_fb': avg_fb[alg],
        }
        for alg in algs
    }

    return vars
