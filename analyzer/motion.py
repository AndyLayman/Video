"""Frame-difference motion signal extraction.

Walks a video sequentially (cheap on most codecs), samples every Nth frame,
and emits a 1-D motion intensity signal restricted to a region of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .video_io import VideoMeta, rel_to_abs_roi


@dataclass
class MotionSignal:
    times: np.ndarray  # shape (N,), seconds
    scores: np.ndarray  # shape (N,), fraction of moving pixels in ROI
    sample_fps: float
    roi: tuple[int, int, int, int]


def extract_motion(
    meta: VideoMeta,
    config: PipelineConfig,
    roi_rel: tuple[float, float, float, float] | None = None,
) -> MotionSignal:
    import cv2  # lazy: heavy native import, not needed by pure-Python callers

    if meta.fps <= 0:
        raise ValueError(f"Invalid source fps for {meta.path}")

    roi_rel = roi_rel or config.pitch_roi_rel
    roi = rel_to_abs_roi(roi_rel, meta.width, meta.height)
    x, y, w, h = roi

    step = max(1, int(round(meta.fps / config.sample_fps)))
    effective_fps = meta.fps / step

    cap = cv2.VideoCapture(str(meta.path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {meta.path}")

    times: list[float] = []
    scores: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                crop = frame[y : y + h, x : x + w]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, binary = cv2.threshold(
                        diff, config.pitch_diff_threshold, 255, cv2.THRESH_BINARY
                    )
                    score = float(np.count_nonzero(binary)) / (w * h)
                    times.append(frame_idx / meta.fps)
                    scores.append(score)
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    return MotionSignal(
        times=np.asarray(times, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float32),
        sample_fps=effective_fps,
        roi=roi,
    )


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    if window <= 1 or signal.size == 0:
        return signal.astype(np.float32, copy=False)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal, kernel, mode="same")


def regional_motion(
    path: str | Path,
    meta: VideoMeta,
    config: PipelineConfig,
    roi_rel: tuple[float, float, float, float],
    start_s: float,
    end_s: float,
) -> MotionSignal:
    """Extract motion only over the [start_s, end_s] window.

    Used by outcome and jersey analyzers that don't need the whole video.
    """
    import cv2

    if meta.fps <= 0:
        raise ValueError("Invalid source fps")
    roi = rel_to_abs_roi(roi_rel, meta.width, meta.height)
    x, y, w, h = roi
    step = max(1, int(round(meta.fps / config.sample_fps)))
    effective_fps = meta.fps / step

    start_frame = max(0, int(start_s * meta.fps))
    end_frame = min(meta.frame_count, int(end_s * meta.fps))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    times: list[float] = []
    scores: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_idx = start_frame
    try:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % step == 0:
                crop = frame[y : y + h, x : x + w]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, binary = cv2.threshold(
                        diff, config.pitch_diff_threshold, 255, cv2.THRESH_BINARY
                    )
                    score = float(np.count_nonzero(binary)) / (w * h)
                    times.append(frame_idx / meta.fps)
                    scores.append(score)
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    return MotionSignal(
        times=np.asarray(times, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float32),
        sample_fps=effective_fps,
        roi=roi,
    )
