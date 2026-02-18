"""
Shared foundations for multi-agent bandit algorithms over erasure channels.

Provides:
    - ``configure_logging`` — one-call logging setup.
    - ``calculate_repetitions`` — compute per-channel delay from erasure probs.
    - Feedback-mode constants (``FEEDBACK_NONE``, …, ``FEEDBACK_ALL``).
    - ``C_CONFIDENCE`` — multiplier for the confidence-region width.
    - ``CommMixin`` — communication-counter bookkeeping (TX / feedback).
    - ``BanditBase(CommMixin)`` — common init, validation, RNG, reset helpers.
"""

import math
import random
from collections import deque
import logging
from typing import Optional, Sequence, Union, Tuple

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure module-level logging. Call once from the application entrypoint.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")


C_CONFIDENCE = 2  # Multiplier in front of the confidence region for SAE


def calculate_repetitions(eps: Union[float, int, Sequence[float]], iters: int, m: Optional[int] = None) -> np.ndarray:
    """
    Calculate repetition counts per channel from erasure probabilities.

    If *m* is provided and *eps* is scalar, broadcast to length *m* to avoid
    downstream indexing errors in multi-agent settings.
    """
    if isinstance(eps, (float, int)) and m is not None:
        eps = [eps] * m
    elif isinstance(eps, (float, int)):
        eps = [eps]
    eps_arr = np.array(eps, dtype=float)
    alphas = np.ones(len(eps_arr))
    for i, eps_val in enumerate(eps_arr):
        if eps_val > 0:
            alphas[i] = np.ceil(2 * np.log(iters) / np.log(1 / eps_val))
    return alphas.astype(int)


# ---------------------------------------------------------------------
# Feedback mode "enum" and shared helpers
# ---------------------------------------------------------------------
FEEDBACK_NONE        = "none"
FEEDBACK_BEACON      = "beacon"        # 1 feedback bit per attempt
FEEDBACK_ACK_SUCCESS = "ack_success"   # feedback only on success (bit == 0)
FEEDBACK_NACK_ERASE  = "nack_erase"    # feedback only on erasure (bit == 1)
FEEDBACK_ALL         = "all"           # always send feedback every round

FEEDBACK_MODES = {
    FEEDBACK_NONE,
    FEEDBACK_BEACON,
    FEEDBACK_ACK_SUCCESS,
    FEEDBACK_NACK_ERASE,
    FEEDBACK_ALL,
}


class CommMixin:
    """
    Mixin that provides communication counter bookkeeping for bandit classes.
    Expects subclasses to define ``m``, ``iters``, and ``feedback_mode``.
    """

    def init_comm_counters(self, feedback_mode: str = FEEDBACK_NONE) -> None:
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"Unknown feedback_mode: {feedback_mode}")
        self.feedback_mode = feedback_mode
        self.tx_count_total = 0
        self.fb_count_total = 0
        self.tx_count_per_agent = np.zeros(self.m, dtype=int)
        self.fb_count_per_agent = np.zeros(self.m, dtype=int)
        self.tx_over_time = np.zeros(self.iters, dtype=int)
        self.fb_over_time = np.zeros(self.iters, dtype=int)
        self.pending_delivery = np.ones(self.m, dtype=bool)
        self.last_delivered_arm = np.full(self.m, -1, dtype=int)

    def reset_comm_counters(self) -> None:
        self.tx_count_total = 0
        self.fb_count_total = 0
        self.tx_count_per_agent[:] = 0
        self.fb_count_per_agent[:] = 0
        self.tx_over_time[:] = 0
        self.fb_over_time[:] = 0
        self.pending_delivery[:] = True
        self.last_delivered_arm[:] = -1

    def reset(
        self,
        mu: Optional[Union[str, Sequence[float]]] = None,
        base_actions=None,
        erasure_seq=None,
    ) -> None:
        """Default reset for bandits that do not override it explicitly."""
        self.reset_base(mu=mu, base_actions=base_actions, erasure_seq=erasure_seq)

    def update_feedback_counts(self, agent_idx: int, bit: int) -> None:
        """
        Update feedback counters according to the current feedback_mode.
        bit = 0 -> success (ACK), bit = 1 -> erasure (NACK).
        """
        mode = getattr(self, "feedback_mode", FEEDBACK_NONE)
        if mode == FEEDBACK_BEACON or mode == FEEDBACK_ALL:
            self.fb_count_total += 1
            self.fb_count_per_agent[agent_idx] += 1
        elif mode == FEEDBACK_ACK_SUCCESS and bit == 0:
            self.fb_count_total += 1
            self.fb_count_per_agent[agent_idx] += 1
        elif mode == FEEDBACK_NACK_ERASE and bit == 1:
            self.fb_count_total += 1
            self.fb_count_per_agent[agent_idx] += 1

    def tx_increment(self, proposed_arms: np.ndarray) -> np.ndarray:
        """
        Compute how many downlink transmissions are sent this round.
        """
        if self.feedback_mode == FEEDBACK_NONE:
            tx_mask = np.ones(self.m, dtype=bool)
        else:
            tx_mask = self.pending_delivery | (proposed_arms != self.last_delivered_arm)
        increment = int(np.sum(tx_mask))
        self.tx_count_total += increment
        self.tx_count_per_agent += tx_mask.astype(int)
        return tx_mask

    def register_delivery(self, proposed_arms: np.ndarray, erasure_bits: np.ndarray, tx_mask: np.ndarray) -> None:
        """Update delivery bookkeeping after observing erasure bits."""
        successes = (erasure_bits == 0) & tx_mask
        still_pending = (~successes) & tx_mask
        self.pending_delivery[tx_mask] = still_pending[tx_mask]
        self.last_delivered_arm[successes] = proposed_arms[successes]

    def sample_erasure_and_update_comm(self, proposed_arms: np.ndarray, erasure_bits: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decide who we transmit to, sample erasures (or use provided), and update TX/FB bookkeeping.
        Returns (tx_mask, erasure_bits_used).
        """
        tx_mask = self.tx_increment(proposed_arms)
        if erasure_bits is None:
            erasure_bits = np.empty(self.m, dtype=int)
            if self.erasure_seq is not None:
                if np.any(self.erasure_index >= self.erasure_seq.shape[0]):
                    erasure_bits[:] = (self.rng.random(self.m) < self.eps).astype(int)
                else:
                    erasure_bits[:] = self.erasure_seq[self.erasure_index, np.arange(self.m)]
                    self.erasure_index += 1
            else:
                erasure_bits[:] = (self.rng.random(self.m) < self.eps).astype(int)
        for idx in range(self.m):
            if tx_mask[idx] or self.feedback_mode == FEEDBACK_ALL:
                self.update_feedback_counts(idx, int(erasure_bits[idx]))
        adj_erasure = erasure_bits.copy()
        adj_erasure[~tx_mask] = 0
        self.register_delivery(proposed_arms, adj_erasure, tx_mask)
        return tx_mask, erasure_bits


class BanditBase(CommMixin):
    """
    Base class providing common initialization, validation, RNG/logging setup,
    and reset helpers for bandit algorithms.
    """

    def __init__(
        self,
        name: str,
        k: int,
        m: int,
        iters: int,
        alphas: Sequence[int],
        var: float = 1.0,
        c: float = 1.0,
        mu: Union[str, Sequence[float]] = "random",
        epsilon: Union[float, Sequence[float]] = 0.0,
        base=None,
        erasure_seq=None,
        feedback_mode: str = FEEDBACK_NONE,
        rng: Optional[Generator] = None,
        verbose: bool = False,
    ):
        self.name = name
        self.k = k
        self.m = m
        self.iters = iters
        self.c = c
        self.variance = var
        self.base = base
        self.rng = rng or np.random.default_rng()
        self.bernoulli = False
        self.verbose = verbose
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.alphas = np.array(alphas)
        self.eps = self._validate_epsilon(epsilon)
        self.mu = self._init_mu(mu)
        self.erasure_seq = self._validate_erasure_seq(erasure_seq)
        self.erasure_index = np.zeros(self.m, dtype=int)

        self._init_logs(base_actions=base)
        self.init_comm_counters(feedback_mode=feedback_mode)

    def _validate_epsilon(self, epsilon: Union[float, Sequence[float]]) -> np.ndarray:
        eps_arr = np.array(epsilon if isinstance(epsilon, (list, np.ndarray)) else [epsilon] * self.m, dtype=float)
        if eps_arr.shape[0] != self.m:
            raise ValueError(f"epsilon length {eps_arr.shape[0]} must match m={self.m}")
        return eps_arr

    def _validate_erasure_seq(self, erasure_seq):
        if erasure_seq is None:
            return None
        arr = np.array(erasure_seq)
        if arr.ndim == 1 and self.m == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < self.iters or arr.shape[1] != self.m:
            raise ValueError(f"erasure_seq shape {arr.shape} must be at least ({self.iters}, {self.m})")
        return arr

    def _init_mu(self, mu: Union[str, Sequence[float]]) -> np.ndarray:
        if isinstance(mu, str) and mu == 'random':
            return self.rng.uniform(0, 1, self.k)
        if isinstance(mu, (list, np.ndarray)):
            mu_arr = np.array(mu, dtype=float)
            if mu_arr.shape[0] != self.k:
                raise ValueError(f"mu length {mu_arr.shape[0]} must match k={self.k}")
            return mu_arr
        raise ValueError(f"Problem with mean initialization: {mu}")

    def _init_logs(self, base_actions=None) -> None:
        self.active_arms = np.ones(self.k, dtype=int)
        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = self.rng.integers(self.k, size=(self.m,)) if self.base is None else np.array(self.base)
        self.pulled_regret = np.zeros(self.m)
        self.pulled_reward = np.zeros(self.m)
        self.regrets = np.zeros((self.iters, self.m))
        self.rewards = np.zeros((self.iters, self.m))
        self.regrets_all = np.zeros(self.iters)
        self.rewards_all = np.zeros(self.iters)

    def reset_base(self, mu: Optional[Union[str, Sequence[float]]] = None, base_actions=None, erasure_seq=None) -> None:
        self.active_arms = np.ones(self.k, dtype=int)
        if base_actions is not None:
            self.pulled_ind = np.array(base_actions)
        else:
            self.pulled_ind = self.rng.integers(self.k, size=(self.m,)) if self.base is None else np.array(self.base)
        self.pulled_regret = np.zeros(self.m)
        self.pulled_reward = np.zeros(self.m)
        self.regrets = np.zeros((self.iters, self.m))
        self.rewards = np.zeros((self.iters, self.m))
        self.regrets_all = np.zeros(self.iters)
        self.rewards_all = np.zeros(self.iters)
        self.erasure_seq = self._validate_erasure_seq(erasure_seq) if erasure_seq is not None else self.erasure_seq
        self.erasure_index = np.zeros(self.m, dtype=int)

        if mu is not None:
            self.mu = self._init_mu(mu)

        self.reset_comm_counters()
