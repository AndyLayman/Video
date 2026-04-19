"""Best-effort jersey-number OCR on the batter region.

EasyOCR is heavy and downloads models on first use; we lazy-load and degrade
gracefully if it isn't installed. The reviewer UI is the source of truth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .pa_segment import PlateAppearance
from .video_io import VideoMeta, rel_to_abs_roi


@dataclass
class JerseyGuess:
    number: str | None
    confidence: float
    samples: list[str]


_reader = None


def _get_reader():
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr  # type: ignore
    except ImportError:
        return None
    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _between_pitch_times(pa: PlateAppearance, n: int) -> list[float]:
    """Pick `n` timestamps where the batter is most likely settled (between pitches)."""
    if not pa.pitches:
        mid = (pa.clip_start_s + pa.clip_end_s) / 2
        return [mid] * n

    candidates: list[float] = []
    # 2s before first pitch
    candidates.append(max(pa.clip_start_s, pa.pitches[0].timestamp_s - 2.0))
    # midpoints between consecutive pitches
    for i in range(len(pa.pitches) - 1):
        a = pa.pitches[i].timestamp_s
        b = pa.pitches[i + 1].timestamp_s
        if b - a > 4.0:
            candidates.append((a + b) / 2)
    if len(candidates) >= n:
        return candidates[:n]
    # Fallback: linearly space across the PA
    extra = np.linspace(pa.clip_start_s + 1.0, pa.clip_end_s - 1.0, n - len(candidates))
    return candidates + list(map(float, extra))


def _digits_only(text: str) -> str | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    # Most baseball numbers are 1-2 digits; reject anything weird.
    if len(digits) > 2:
        digits = digits[:2]
    return digits


def detect_jersey(
    video_path: Path,
    meta: VideoMeta,
    pa: PlateAppearance,
    config: PipelineConfig,
) -> JerseyGuess:
    import cv2

    reader = _get_reader()
    if reader is None:
        return JerseyGuess(number=None, confidence=0.0, samples=[])

    sample_times = _between_pitch_times(pa, config.jersey_ocr_samples)
    roi = rel_to_abs_roi(config.batter_roi_rel, meta.width, meta.height)
    x, y, w, h = roi

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return JerseyGuess(None, 0.0, [])

    readings: list[str] = []
    try:
        for t in sample_times:
            frame_idx = int(t * meta.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            crop = frame[y : y + h, x : x + w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Upscale to help OCR on small numbers.
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            try:
                results = reader.readtext(gray, allowlist="0123456789", detail=0)
            except Exception:
                results = []
            for r in results:
                d = _digits_only(str(r))
                if d:
                    readings.append(d)
    finally:
        cap.release()

    if not readings:
        return JerseyGuess(None, 0.0, [])
    counter = Counter(readings)
    number, freq = counter.most_common(1)[0]
    confidence = min(1.0, freq / max(1, len(readings)))
    return JerseyGuess(number=number, confidence=round(confidence, 2), samples=readings)
