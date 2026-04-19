"""Best-effort outcome inference from post-pitch motion.

Without an overlay or trained model, this is intentionally rough. It produces
a starting guess for the human reviewer, never a confident verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .motion import smooth
from .pa_segment import PlateAppearance
from .video_io import VideoMeta

# Rough mapping from ball direction in the frame to a likely fielder.
# Frame is from the backstop, so left side of frame = third base side.
ZONE_TO_POSITION = {
    "far_left": "3B",
    "left": "SS",
    "center": "P/2B",
    "right": "1B",
    "far_right": "1B",
    "left_outfield": "LF",
    "center_outfield": "CF",
    "right_outfield": "RF",
}


@dataclass
class OutcomeGuess:
    in_play: bool
    confidence: float  # 0..1
    field_zone: str | None  # one of ZONE_TO_POSITION keys
    likely_position: str | None
    notes: str = ""


def _zone_for_column(col_frac: float, vertical_frac: float) -> str:
    """Map an (x, y) location in the frame to a coarse field zone.

    col_frac in [0,1]: 0 = left edge of frame, 1 = right edge.
    vertical_frac in [0,1]: 0 = top of frame (deep), 1 = bottom (close).
    """
    deep = vertical_frac < 0.35
    if deep:
        if col_frac < 0.33:
            return "left_outfield"
        if col_frac < 0.66:
            return "center_outfield"
        return "right_outfield"
    if col_frac < 0.18:
        return "far_left"
    if col_frac < 0.42:
        return "left"
    if col_frac < 0.58:
        return "center"
    if col_frac < 0.82:
        return "right"
    return "far_right"


def analyze_outcome(
    video_path: Path,
    meta: VideoMeta,
    pa: PlateAppearance,
    config: PipelineConfig,
) -> OutcomeGuess:
    """Look at motion in the field after the final pitch to guess the outcome."""
    import cv2

    if not pa.pitches:
        return OutcomeGuess(False, 0.0, None, None, "no pitches in PA")

    start_s = pa.last_pitch_s + 0.4
    end_s = min(pa.clip_end_s, start_s + config.outcome_window_s)
    if end_s <= start_s:
        return OutcomeGuess(False, 0.0, None, None, "empty post-pitch window")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return OutcomeGuess(False, 0.0, None, None, "video unreadable")

    step = max(1, int(round(meta.fps / config.sample_fps)))
    start_frame = int(start_s * meta.fps)
    end_frame = int(end_s * meta.fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    motion_per_frame: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_idx = start_frame
    try:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, binary = cv2.threshold(
                        diff, config.pitch_diff_threshold, 255, cv2.THRESH_BINARY
                    )
                    nz = float(np.count_nonzero(binary))
                    motion_per_frame.append(nz / binary.size)
                    if nz > 0:
                        ys, xs = np.nonzero(binary)
                        centroid_x.append(float(xs.mean()) / binary.shape[1])
                        centroid_y.append(float(ys.mean()) / binary.shape[0])
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    if not motion_per_frame:
        return OutcomeGuess(False, 0.0, None, None, "no motion samples")

    arr = np.asarray(motion_per_frame, dtype=np.float32)
    smoothed = smooth(arr, max(3, int(config.sample_fps * 0.4)))

    baseline = float(np.median(smoothed))
    peak = float(smoothed.max())
    sustained_threshold = max(baseline * 2.5, 0.01)
    sustained_samples = int(np.sum(smoothed > sustained_threshold))
    sustained_dur = sustained_samples / config.sample_fps

    in_play = sustained_dur >= config.in_play_motion_dur_s and peak > 0.02

    field_zone: str | None = None
    likely_position: str | None = None
    confidence = 0.0
    notes_parts: list[str] = [
        f"sustained={sustained_dur:.1f}s",
        f"peak={peak:.3f}",
        f"baseline={baseline:.3f}",
    ]

    if in_play and centroid_x:
        x_avg = float(np.mean(centroid_x))
        y_avg = float(np.mean(centroid_y))
        field_zone = _zone_for_column(x_avg, y_avg)
        likely_position = ZONE_TO_POSITION.get(field_zone)
        # Confidence rises with how clear the motion signal is over baseline.
        confidence = max(0.0, min(1.0, (peak - baseline) / 0.05))
        notes_parts.append(f"centroid=({x_avg:.2f},{y_avg:.2f})")

    return OutcomeGuess(
        in_play=in_play,
        confidence=round(confidence, 2),
        field_zone=field_zone,
        likely_position=likely_position,
        notes="; ".join(notes_parts),
    )
