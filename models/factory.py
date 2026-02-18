"""
models.factory — Bandit algorithm factory function.
"""

import logging

from models.base import (
    calculate_repetitions,
    FEEDBACK_NONE,
)
from models.batch_sp2 import BatchSP2
from models.batch_sp2_erasure import BatchSP2Erasure
from models.batch_sp2_rrr import BatchSP2RRR
from models.batch_sgreedy import BatchSGreedy
from models.batch_tpg import BatchTPG
from models.batch_tpg_old import BatchTPGOld
from models.batch_tpg_new import BatchTPGNew
from models.batch_sp2_simplified import BatchSP2Simplified
from models.ucb_ma import ucb_ma
from models.lsae_horizontal import LSAE_ma_hor
from models.lsae_vertical import LSAE_ma_ver
from models.vanilla_sae import Vanilla_SAE_ma

logger = logging.getLogger(__name__)


def init_bandit(alg_name, rep=True, mode='scheduled', iters=10000, k=10, m=1, mu='random',
                eps=0, var=1, delta=0.01, c=1, base_actions=None,
                feedback_mode=FEEDBACK_NONE, rng=None):
    """
    Factory function for bandit algorithms.

    Supports both legacy AISTATS algorithms (UCB, Vanilla_SAE, LSAE horizontal/vertical)
    and new JSAIT algorithms (scheduled, feedback, tpg, etc.).
    """
    alphas = calculate_repetitions(eps, iters, m=m)
    logger.debug("repetitions=%s eps=%s", alphas, eps)
    logger.debug("Algo=%s mode=%s feedback=%s", alg_name, mode, feedback_mode)

    # --- Legacy AISTATS routes ---
    if alg_name.upper() == 'UCB':
        return ucb_ma(k, m, iters, alphas=alphas, var=var, mu=mu, epsilon=eps, c=c,
                      base=base_actions, feedback_mode=feedback_mode, rng=rng)

    if alg_name.upper() == 'SAE':
        if not rep:
            return Vanilla_SAE_ma(k, m, iters=iters, alphas=alphas, var=var, mu=mu,
                                  epsilon=eps, c=c, base=base_actions,
                                  feedback_mode=feedback_mode, rng=rng)

        mode_l = mode.lower()

        # Legacy assignment modes
        if mode_l in ('horizontal', 'weighted'):
            return LSAE_ma_hor(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                               base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'vertical':
            return LSAE_ma_ver(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                               base=base_actions, feedback_mode=feedback_mode, rng=rng)

        # --- New JSAIT routes ---
        elif mode_l == 'scheduled':
            return BatchSP2(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                            base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'feedback':
            return BatchSP2Erasure(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                                   base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'rrr':
            return BatchSP2RRR(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                               base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'greedy':
            return BatchSGreedy(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                                base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'tpg-old':
            return BatchTPGOld(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                               base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'tpg':
            return BatchTPG(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                            base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l == 'tpg-new':
            return BatchTPGNew(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                               base=base_actions, feedback_mode=feedback_mode, rng=rng)
        elif mode_l in ('tpg-simplified', 'tpg-simple', 'simple'):
            return BatchSP2Simplified(k, m, iters, alphas, var=var, mu=mu, epsilon=eps, c=c,
                                      base=base_actions, feedback_mode=feedback_mode, rng=rng)
        else:
            logger.error('Error in the mode for SAE with repetition, mode: %s', mode)
            return
    else:
        logger.error('Unknown algorithm name: %s', alg_name)
        return None
