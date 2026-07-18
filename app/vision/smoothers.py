"""
app/vision/smoothers.py — Temporal smoothing utilities for vision analysis.

  EMASmoother         — Per-channel exponential moving average for blendshapes.
  SlidingWindowVoter  — Majority-vote over a rolling window to prevent flickering.
"""

from collections import deque

import numpy as np


class EMASmoother:
    """
    Applies Exponential Moving Average smoothing to a vector of N values.
    alpha=0.4 → responsive but smooth (lower = more smoothing, more lag).

    Formula: smoothed[t] = alpha * new[t] + (1 - alpha) * smoothed[t-1]
    """

    def __init__(self, size: int, alpha: float = 0.4):
        self._alpha = alpha
        self._prev  = None
        self._size  = size

    def update(self, values: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = values.copy()
            return values
        smoothed   = self._alpha * values + (1.0 - self._alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


class SlidingWindowVoter:
    """
    Maintains a rolling window of expression predictions.
    Returns the majority vote — prevents single-frame noise from
    changing the reported expression.

    window_size=5, min_votes=3 → expression changes only if 3/5 frames agree.
    """

    def __init__(self, window_size: int = 5, min_votes: int = 3):
        self._window    = deque(maxlen=window_size)
        self._min_votes = min_votes
        self._current   = "neutral"

    def vote(self, expression: str) -> str:
        self._window.append(expression)
        if len(self._window) < self._min_votes:
            return expression  # not enough history yet

        from collections import Counter
        counts = Counter(self._window)
        top, top_count = counts.most_common(1)[0]
        if top_count >= self._min_votes:
            self._current = top
        return self._current

    def reset(self):
        self._window.clear()
        self._current = "neutral"
