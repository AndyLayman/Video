"""Detect pitch events as rising-edge crossings on the motion signal.

A "pitch" looks like: quiet → brief spike → quiet. A ball in play or a
sustained fielding sequence stays *above* threshold for a long time and
should count as exactly one event, not many. Rising-edge detection with a
post-detection cooldown gives us that behavior cleanly.
"""

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
    """O(n*window) rolling median + MAD. Fine for our sample sizes."""
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

    roll_window = max(5, int(round(config.pitch_roll_window_s * motion.sample_fps)))
    medians, mads = _rolling_median_mad(smoothed, roll_window)
    mads = np.maximum(mads, 1e-6)
    thresholds = medians + config.pitch_peak_sigma * mads

    above = smoothed > thresholds
    # Enforce absolute floor so ambient noise in quiet stretches (e.g. kids
    # milling around between batters) never qualifies as a pitch.
    above = above & (smoothed > config.pitch_abs_floor)

    cooldown_samples = max(1, int(round(config.pitch_min_gap_s * motion.sample_fps)))

    pitches: list[PitchEvent] = []
    prev = False
    cooldown_until = -1
    # Find the peak value within the "above" run that starts each rising edge,
    # report that peak's timestamp (more stable than reporting the edge sample).
    run_start = -1
    for i in range(smoothed.size):
        a = bool(above[i])
        if a and not prev and i >= cooldown_until:
            run_start = i
        if not a and prev and run_start >= 0:
            seg = smoothed[run_start : i]
            peak_offset = int(np.argmax(seg))
            peak_idx = run_start + peak_offset
            pitches.append(
                PitchEvent(
                    timestamp_s=float(motion.times[peak_idx]),
                    score=float(smoothed[peak_idx]),
                )
            )
            cooldown_until = peak_idx + cooldown_samples
            run_start = -1
        prev = a
    # Close an open run at the end of the signal.
    if run_start >= 0 and run_start >= cooldown_until:
        seg = smoothed[run_start:]
        peak_offset = int(np.argmax(seg))
        peak_idx = run_start + peak_offset
        pitches.append(
            PitchEvent(
                timestamp_s=float(motion.times[peak_idx]),
                score=float(smoothed[peak_idx]),
            )
        )

    return pitches
