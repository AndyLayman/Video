"""Detect pitch events as motion peaks in the pitcher->plate corridor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .motion import MotionSignal, smooth


@dataclass
class PitchEvent:
    timestamp_s: float
    score: float


def _rolling_median_mad(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """O(n*window) rolling median + MAD using a sliding slice.

    Fine for our sample sizes (a few thousand). Avoids a scipy dependency.
    """
    window = max(3, window)
    n = x.size
    half = window // 2
    medians = np.empty(n, dtype=np.float32)
    mads = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        seg = x[start:end]
        m = float(np.median(seg))
        medians[i] = m
        mads[i] = float(np.median(np.abs(seg - m)))
    return medians, mads


def detect_pitches(motion: MotionSignal, config: PipelineConfig) -> list[PitchEvent]:
    if motion.scores.size == 0:
        return []

    smooth_window = max(1, int(round(config.pitch_smooth_s * motion.sample_fps)))
    smoothed = smooth(motion.scores, smooth_window)

    # Rolling baseline so quiet sections of the video aren't swamped by
    # the global median (which would be inflated by busy PAs).
    roll_window = max(5, int(round(config.pitch_roll_window_s * motion.sample_fps)))
    medians, mads = _rolling_median_mad(smoothed, roll_window)
    mads = np.maximum(mads, 1e-6)
    thresholds = medians + config.pitch_peak_sigma * mads

    min_gap_samples = max(1, int(round(config.pitch_min_gap_s * motion.sample_fps)))
    # Require peaks to be the dominant local maximum in a small window so
    # we don't shatter a single event into multiple "pitches".
    local_max_window = max(1, int(round(config.pitch_min_gap_s * motion.sample_fps * 0.5)))

    peaks: list[PitchEvent] = []
    last_peak_idx = -min_gap_samples - 1
    n = smoothed.size

    for i in range(1, n - 1):
        s = smoothed[i]
        if s <= thresholds[i]:
            continue
        if not (s >= smoothed[i - 1] and s >= smoothed[i + 1]):
            continue
        lo = max(0, i - local_max_window)
        hi = min(n, i + local_max_window + 1)
        if smoothed[lo:hi].max() > s:
            continue
        if i - last_peak_idx < min_gap_samples:
            if peaks and s > peaks[-1].score:
                peaks[-1] = PitchEvent(timestamp_s=float(motion.times[i]), score=float(s))
                last_peak_idx = i
            continue
        peaks.append(PitchEvent(timestamp_s=float(motion.times[i]), score=float(s)))
        last_peak_idx = i

    return peaks
