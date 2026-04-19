"""Thin wrappers around OpenCV video I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def probe(path: str | Path) -> VideoMeta:
    import cv2

    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {p}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return VideoMeta(path=p, fps=fps, frame_count=frame_count, width=width, height=height)


def rel_to_abs_roi(
    roi_rel: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x = max(0, int(roi_rel[0] * width))
    y = max(0, int(roi_rel[1] * height))
    w = max(1, int(roi_rel[2] * width))
    h = max(1, int(roi_rel[3] * height))
    w = min(w, width - x)
    h = min(h, height - y)
    return x, y, w, h
