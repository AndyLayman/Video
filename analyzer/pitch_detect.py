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


def detect_pitches(motion: MotionSignal, config: PipelineConfig) -> list[PitchEvent]:
    if motion.scores.size == 0:
        return []

    smooth_window = max(1, int(round(config.pitch_smooth_s * motion.sample_fps)))
    smoothed = smooth(motion.scores, smooth_window)

    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    if mad <= 0:
        mad = float(np.std(smoothed)) or 1e-6
    threshold = median + config.pitch_peak_sigma * mad

    min_gap_samples = max(1, int(round(config.pitch_min_gap_s * motion.sample_fps)))

    peaks: list[PitchEvent] = []
    last_peak_idx = -min_gap_samples - 1
    n = smoothed.size

    for i in range(1, n - 1):
        s = smoothed[i]
        if s <= threshold:
            continue
        if not (s >= smoothed[i - 1] and s >= smoothed[i + 1]):
            continue
        if i - last_peak_idx < min_gap_samples:
            # Keep the larger of the two competing peaks within the gap
            if peaks and s > peaks[-1].score:
                peaks[-1] = PitchEvent(timestamp_s=float(motion.times[i]), score=float(s))
                last_peak_idx = i
            continue
        peaks.append(PitchEvent(timestamp_s=float(motion.times[i]), score=float(s)))
        last_peak_idx = i

    return peaks
