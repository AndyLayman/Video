"""Tunable defaults for the analysis pipeline.

All numbers here are starting heuristics for a fixed-backstop GameChanger
camera. Override per-video via the CLI when needed.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # Sampling
    sample_fps: float = 15.0  # frames/sec to analyze for motion

    # Pitch detection
    pitch_roi_rel: tuple[float, float, float, float] = (0.33, 0.05, 0.34, 0.60)
    """ROI (x, y, w, h) as fractions of frame size. Default: middle vertical
    strip from the pitcher's mound area down toward the plate."""
    pitch_diff_threshold: int = 25
    """Per-pixel intensity delta to count as motion."""
    pitch_peak_sigma: float = 4.0
    """Adaptive threshold = median + sigma * MAD."""
    pitch_min_gap_s: float = 5.0
    """Minimum seconds between successive pitch events."""
    pitch_smooth_s: float = 0.3
    """Boxcar smoothing window for the motion signal."""

    # Plate appearance segmentation
    pa_max_gap_s: float = 35.0
    """Gap between pitches above which we declare a new PA."""
    pa_lead_s: float = 6.0
    """Seconds before the first pitch of a PA included in the clip."""
    pa_trail_s: float = 8.0
    """Seconds after the last pitch of a PA included in the clip."""

    # Outcome heuristics
    outcome_window_s: float = 6.0
    """Window after the last pitch to inspect for in-play motion."""
    in_play_motion_dur_s: float = 1.5
    """Sustained motion duration above baseline that suggests ball in play."""

    # Jersey OCR
    batter_roi_rel: tuple[float, float, float, float] = (0.40, 0.55, 0.20, 0.30)
    """ROI for the batter's back. Override per-camera."""
    jersey_ocr_samples: int = 8
    """Frames sampled per PA for jersey OCR."""

    # Clip extraction
    ffmpeg_bin: str = "ffmpeg"
    use_stream_copy: bool = True
    """If True, ffmpeg uses -c copy (fast, may snap to keyframes)."""


DEFAULT_CONFIG = PipelineConfig()
