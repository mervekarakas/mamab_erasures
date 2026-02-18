"""
helper_methods — Backward-compatibility shim.

All functionality has moved to dedicated modules:
  utils.py          Erasure sequence generation
  models/factory.py Bandit factory (init_bandit)
  runners.py        Episode runners
  plotting/utils.py plot_and_save
"""

from utils import *  # noqa: F401,F403
from models.factory import init_bandit  # noqa: F401
from runners import _run_episode_worker, run_episodes, run_episodes_with_same_erasures  # noqa: F401
from plotting.utils import plot_and_save  # noqa: F401
