"""End-to-end orchestration: video in, clips + analysis.json out."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .clip import cut_clip, ensure_ffmpeg
from .config import DEFAULT_CONFIG, PipelineConfig
from .jersey_ocr import detect_jersey
from .motion import extract_motion
from .outcome import analyze_outcome
from .pa_segment import group_into_pas
from .pitch_detect import detect_pitches
from .video_io import probe

__all__ = [
    "ANALYSIS_FILENAME",
    "analyze_inning",
    "load_analysis",
    "save_analysis",
    "recut_from_markers",
]

ANALYSIS_FILENAME = "analysis.json"
SCHEMA_VERSION = 1


def _format_ts(seconds: float) -> str:
    s = max(0.0, seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def analyze_inning(
    video_path: str | Path,
    out_dir: str | Path,
    inning: str | None = None,
    config: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
    debug_motion_csv: str | Path | None = None,
) -> dict:
    """Run the full pipeline. Returns the analysis dict (also written to disk)."""
    config = config or DEFAULT_CONFIG
    log = progress or (lambda msg: None)

    src = Path(video_path).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = Path(out_dir).resolve()
    clips_dir = out_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    ensure_ffmpeg(config)

    log(f"Probing {src.name}")
    meta = probe(src)
    log(
        f"  {meta.width}x{meta.height} @ {meta.fps:.2f}fps, "
        f"{meta.duration_s:.1f}s ({meta.frame_count} frames)"
    )

    log("Extracting motion signal")
    motion = extract_motion(meta, config)

    if debug_motion_csv is not None:
        from .debug import dump_motion_csv

        dump_motion_csv(motion.times, motion.scores, Path(debug_motion_csv))
        log(f"  wrote motion CSV to {debug_motion_csv}")

    import numpy as np

    if motion.scores.size:
        med = float(np.median(motion.scores))
        mad = float(np.median(np.abs(motion.scores - med)))
        log(
            f"  motion stats (global): median={med:.5f} mad={mad:.5f} "
            f"(threshold is now rolling, see pitch_roll_window_s)"
        )

    log("Detecting pitch events")
    pitches = detect_pitches(motion, config)
    log(f"  found {len(pitches)} pitch candidates")

    log("Grouping into plate appearances")
    pas = group_into_pas(pitches, config, meta.duration_s)
    log(f"  segmented into {len(pas)} PAs")

    # Copy original to out_dir/full.<ext> so it travels with the analysis.
    full_dst = out_dir / f"full{src.suffix.lower()}"
    if not full_dst.exists() or full_dst.stat().st_size != src.stat().st_size:
        log(f"Copying source video → {full_dst.name}")
        shutil.copy2(src, full_dst)

    pa_records: list[dict] = []
    for pa in pas:
        clip_name = f"pa_{pa.index:02d}.mp4"
        clip_path = clips_dir / clip_name
        log(
            f"PA {pa.index}: {pa.pitch_count} pitches, "
            f"{_format_ts(pa.clip_start_s)}–{_format_ts(pa.clip_end_s)} → {clip_name}"
        )
        try:
            cut_clip(src, clip_path, pa.clip_start_s, pa.clip_end_s, config)
        except Exception as e:
            log(f"  clip failed: {e}")
            continue

        log("  guessing outcome")
        outcome = analyze_outcome(src, meta, pa, config)
        log(
            f"    in_play={outcome.in_play} zone={outcome.field_zone} "
            f"pos={outcome.likely_position} conf={outcome.confidence}"
        )

        log("  attempting jersey OCR")
        jersey = detect_jersey(src, meta, pa, config)
        if jersey.number:
            log(f"    jersey={jersey.number} conf={jersey.confidence}")
        else:
            log("    jersey=unknown")

        pa_records.append(
            {
                "index": pa.index,
                "clip": f"clips/{clip_name}",
                "start_s": round(pa.clip_start_s, 3),
                "end_s": round(pa.clip_end_s, 3),
                "start_ts": _format_ts(pa.clip_start_s),
                "end_ts": _format_ts(pa.clip_end_s),
                "pitch_count": pa.pitch_count,
                "pitches_s": [round(p.timestamp_s, 3) for p in pa.pitches],
                "auto": {
                    "jersey": jersey.number,
                    "jersey_confidence": jersey.confidence,
                    "jersey_samples": jersey.samples,
                    "in_play": outcome.in_play,
                    "field_zone": outcome.field_zone,
                    "likely_position": outcome.likely_position,
                    "outcome_confidence": outcome.confidence,
                    "outcome_notes": outcome.notes,
                },
                "edits": {
                    "jersey": None,
                    "fielding_position": None,
                    "out": None,
                    "outcome": None,
                    "notes": None,
                    "reviewed": False,
                },
            }
        )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "source_video": str(src),
        "stored_video": full_dst.name,
        "inning": inning,
        "video": {
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
            "duration_s": round(meta.duration_s, 3),
        },
        "config": asdict(config),
        "plate_appearances": pa_records,
    }

    out_path = out_dir / ANALYSIS_FILENAME
    out_path.write_text(json.dumps(analysis, indent=2))
    log(f"Wrote {out_path}")
    return analysis


def load_analysis(out_dir: str | Path) -> dict:
    p = Path(out_dir) / ANALYSIS_FILENAME
    return json.loads(p.read_text())


def save_analysis(out_dir: str | Path, analysis: dict) -> None:
    p = Path(out_dir) / ANALYSIS_FILENAME
    p.write_text(json.dumps(analysis, indent=2))


def _blank_edits() -> dict:
    return {
        "jersey": None,
        "fielding_position": None,
        "out": None,
        "outcome": None,
        "notes": None,
        "reviewed": False,
        "pitch_count": None,
    }


def recut_from_markers(
    inning_dir: str | Path,
    markers: list[dict],
    config: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Replace the clips directory and PAs in analysis.json from user markers.

    Each marker is `{"start_s": float}`. PA end time = next marker's start, or
    the video duration for the last one. Any existing `edits` block is
    preserved by PA index so a user's prior corrections don't get wiped on
    re-cut.
    """
    config = config or DEFAULT_CONFIG
    log = progress or (lambda msg: None)
    inning_dir = Path(inning_dir)
    analysis = load_analysis(inning_dir)

    stored = analysis.get("stored_video") or "full.mp4"
    src = inning_dir / stored
    if not src.exists():
        raise FileNotFoundError(f"source video missing: {src}")

    duration_s = float(analysis["video"]["duration_s"])

    sorted_markers = sorted(
        [m for m in markers if m is not None],
        key=lambda m: float(m["start_s"]),
    )

    clips_dir = inning_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("pa_*.mp4"):
        old.unlink()

    ensure_ffmpeg(config)

    # Preserve user edits keyed by PA index.
    old_edits: dict[int, dict] = {
        pa["index"]: pa.get("edits") or _blank_edits()
        for pa in analysis.get("plate_appearances", [])
    }

    new_pas: list[dict] = []
    for i, m in enumerate(sorted_markers):
        start_s = max(0.0, float(m["start_s"]))
        end_s = (
            float(sorted_markers[i + 1]["start_s"])
            if i + 1 < len(sorted_markers)
            else duration_s
        )
        end_s = min(duration_s, end_s)
        if end_s <= start_s + 0.5:
            log(f"skip marker at {start_s:.2f}s (zero-length)")
            continue
        idx = len(new_pas) + 1
        clip_name = f"pa_{idx:02d}.mp4"
        clip_path = clips_dir / clip_name
        log(f"PA {idx}: {start_s:.2f}-{end_s:.2f}s -> {clip_name}")
        cut_clip(src, clip_path, start_s, end_s, config)
        new_pas.append(
            {
                "index": idx,
                "clip": f"clips/{clip_name}",
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "start_ts": _format_ts(start_s),
                "end_ts": _format_ts(end_s),
                "pitch_count": 0,
                "pitches_s": [],
                "auto": {
                    "jersey": None,
                    "jersey_confidence": 0.0,
                    "jersey_samples": [],
                    "in_play": False,
                    "field_zone": None,
                    "likely_position": None,
                    "outcome_confidence": 0.0,
                    "outcome_notes": "manual marker",
                },
                "edits": old_edits.get(idx, _blank_edits()),
            }
        )

    analysis["plate_appearances"] = new_pas
    analysis["manual_markers"] = True
    save_analysis(inning_dir, analysis)
    log(f"wrote {len(new_pas)} PAs to {inning_dir / ANALYSIS_FILENAME}")
    return analysis
