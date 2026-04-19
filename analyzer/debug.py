"""Diagnostics: visualize ROIs and motion signals to help tune the pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .video_io import VideoMeta, probe, rel_to_abs_roi


def save_roi_overlay(video_path: Path, out_png: Path, config: PipelineConfig) -> Path:
    """Save a frame from the middle of the video with the configured ROIs drawn on it."""
    import cv2

    meta = probe(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    try:
        target = max(0, meta.frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("could not read sample frame")
    finally:
        cap.release()

    pitch = rel_to_abs_roi(config.pitch_roi_rel, meta.width, meta.height)
    batter = rel_to_abs_roi(config.batter_roi_rel, meta.width, meta.height)

    px, py, pw, ph = pitch
    bx, by, bw, bh = batter
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 6)
    cv2.putText(
        frame,
        "pitch ROI",
        (px + 10, py + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        4,
    )
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 255), 6)
    cv2.putText(
        frame,
        "batter ROI",
        (bx + 10, by + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 200, 255),
        4,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)
    return out_png


def dump_motion_csv(times: np.ndarray, scores: np.ndarray, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("t_s,score\n")
        for t, s in zip(times, scores):
            f.write(f"{t:.4f},{s:.6f}\n")
    return out_csv
