"""
models — Multi-agent multi-armed bandit algorithms for erasure channels.

JSAIT 2026 algorithms:
  BatchSP2            Offline scheduled batch pulls, no feedback (AISTATS 2024 baseline)
  BatchSP2Erasure     SP2 with erasure feedback and stack-based delivery ("SP2-Feedback")
  BatchTPG            Two-Phase Greedy — main JSAIT 2026 algorithm
  BatchTPGNew         TPG variant that guards delivered arms from takeover
  BatchTPGOld         TPG variant with forcible unassignment (superseded)
  BatchSP2RRR         Random round-robin scheduling baseline
  BatchSGreedy        Greedy per-round arm selection baseline
  BatchSP2Simplified  Simplified TPG with pointer-based scheduling

Legacy AISTATS algorithms:
  ucb_ma              Multi-Agent UCB
  LSAE_ma_hor         Successive Arm Elimination with horizontal assignment
  LSAE_ma_ver         Successive Arm Elimination with vertical assignment
  Vanilla_SAE_ma      Successive Arm Elimination without repetition
"""

from models.base import (
    CommMixin,
    BanditBase,
    calculate_repetitions,
    C_CONFIDENCE,
    FEEDBACK_NONE,
    FEEDBACK_BEACON,
    FEEDBACK_ACK_SUCCESS,
    FEEDBACK_NACK_ERASE,
    FEEDBACK_ALL,
)

from models.batch_sp2 import BatchSP2
from models.batch_sp2_erasure import BatchSP2Erasure
from models.batch_tpg import BatchTPG
from models.batch_tpg_old import BatchTPGOld
from models.batch_tpg_new import BatchTPGNew
from models.batch_sp2_rrr import BatchSP2RRR
from models.batch_sgreedy import BatchSGreedy
from models.batch_sp2_simplified import BatchSP2Simplified

from models.ucb_ma import ucb_ma
from models.lsae_horizontal import LSAE_ma_hor
from models.lsae_vertical import LSAE_ma_ver
from models.vanilla_sae import Vanilla_SAE_ma

from models.factory import init_bandit

__all__ = [
    # Base
    "CommMixin",
    "BanditBase",
    "calculate_repetitions",
    "C_CONFIDENCE",
    "FEEDBACK_NONE",
    "FEEDBACK_BEACON",
    "FEEDBACK_ACK_SUCCESS",
    "FEEDBACK_NACK_ERASE",
    "FEEDBACK_ALL",
    # JSAIT algorithms
    "BatchSP2",
    "BatchSP2Erasure",
    "BatchTPG",
    "BatchTPGOld",
    "BatchTPGNew",
    "BatchSP2RRR",
    "BatchSGreedy",
    "BatchSP2Simplified",
    # Legacy algorithms
    "ucb_ma",
    "LSAE_ma_hor",
    "LSAE_ma_ver",
    "Vanilla_SAE_ma",
    # Factory
    "init_bandit",
]
