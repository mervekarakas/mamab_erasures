"""
utils — Erasure sequence generation for multi-agent bandit simulations.
"""

import random
import numpy as np


def generate_erasure_sequences(iters, episodes, eps):
    seqs = []
    for _ in range(episodes):
        seq = np.random.choice([0, 1], size=iters, p=[1 - eps, eps])
        seqs.append(seq)
    return seqs


def generate_varying_erasure_sequence(iters, episodes, start_eps=0.1, end_eps=0.9, breakpoint_frac=0.2, mode='linear'):
    """
    Generates a list of predetermined erasure sequences (one per episode) of length 'iters'.
    """
    seqs = []

    if mode.lower() == 'linear':
        eps_schedule = np.linspace(start_eps, end_eps, iters)
    elif mode.lower() == 'abrupt':
        T_break = int(iters * breakpoint_frac)
        if T_break < 1:
            T_break = 1
        linear_schedule = np.linspace(start_eps, end_eps, T_break)
        constant_schedule = np.full(iters - T_break, end_eps)
        eps_schedule = np.concatenate((linear_schedule, constant_schedule))
    else:
        raise ValueError("Unknown mode. Please use 'linear' or 'abrupt'.")

    for _ in range(episodes):
        erasure_seq = np.array([1 if random.random() < eps else 0 for eps in eps_schedule])
        seqs.append(erasure_seq)

    return seqs


def generate_abrupt_erasure_sequence(iters, episodes, breakpoints, epsilons):
    """
    Generates a predetermined erasure sequence of length 'iters' with abrupt changes in channel quality.
    """
    seqs = []
    for _ in range(episodes):
        sequence = []
        start = 0
        for bp, eps in zip(breakpoints, epsilons):
            segment_length = bp - start
            segment = np.random.choice([0, 1], size=segment_length, p=[1 - eps, eps])
            sequence.extend(segment)
            start = bp
        if start < iters:
            segment_length = iters - start
            segment = np.random.choice([0, 1], size=segment_length, p=[1 - epsilons[-1], epsilons[-1]])
            sequence.extend(segment)
        seqs.append(np.array(sequence))
    return seqs


def generate_erasure_sequence_multi(iters, m, eps):
    if np.isscalar(eps):
        eps = np.full(m, eps)
    if len(eps) != m:
        raise ValueError(f"eps length ({len(eps)}) must match m ({m})")
    seq = np.zeros((iters, m), dtype=int)
    for t in range(iters):
        for mm in range(m):
            if np.random.rand() < eps[mm]:
                seq[t, mm] = 1
    return seq
