from __future__ import annotations

import numpy as np


class RunningSearchStatistics:
    """Per-complexity frequency tracker with sliding window.

    Mirrors Julia's ``SymbolicRegression.AdaptiveParsimony.RunningSearchStatistics``.
    """

    __slots__ = ("window_size", "frequencies", "normalized_frequencies")

    def __init__(self, maxsize: int, window_size: int = 100000) -> None:
        self.window_size = window_size
        self.frequencies = np.ones(maxsize, dtype=np.float64)
        self.normalized_frequencies = np.ones(maxsize, dtype=np.float64) / maxsize

    def update_frequencies(self, size: int) -> None:
        if 0 < size <= len(self.frequencies):
            self.frequencies[size - 1] += 1.0

    def move_window(self) -> None:
        freqs = self.frequencies
        window = float(self.window_size)
        cur_sum = float(np.sum(freqs))
        if cur_sum <= window:
            return

        diff = cur_sum - window
        smallest_allowed = 1.0
        max_loops = 1000
        num_loops = 0

        while diff > 0.0 and num_loops < max_loops:
            mask = freqs > smallest_allowed
            n_remaining = int(np.sum(mask))
            if n_remaining == 0:
                break
            above_min = freqs[mask]
            amount = min(diff / n_remaining, float(np.min(above_min)) - smallest_allowed)
            if amount < 1e-6:
                break
            freqs[mask] -= amount
            diff -= amount * n_remaining
            num_loops += 1

    def normalize_frequencies(self) -> None:
        total = float(np.sum(self.frequencies))
        if total > 0.0:
            self.normalized_frequencies = self.frequencies / total
