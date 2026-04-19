"""CLI entrypoint: analyze an inning video into per-PA clips and JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import analyze_inning


def _parse_roi(s: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x,y,w,h' as fractions in [0,1]")
    if any(p < 0 or p > 1 for p in parts):
        raise argparse.ArgumentTypeError("ROI fractions must be in [0,1]")
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze",
        description="Segment a GameChanger-style inning video into per-batter clips.",
    )
    p.add_argument("video", type=Path, help="Path to the inning video (mp4/mov).")
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="Output directory for clips and analysis.json.",
    )
    p.add_argument(
        "--inning",
        type=str,
        default=None,
        help='Inning label to embed in analysis.json (e.g. "T3" or "B5").',
    )
    p.add_argument(
        "--pitch-roi",
        type=_parse_roi,
        default=None,
        help=(
            "Pitcher→plate ROI as 'x,y,w,h' fractions of the frame. "
            f"Default: {DEFAULT_CONFIG.pitch_roi_rel}"
        ),
    )
    p.add_argument(
        "--batter-roi",
        type=_parse_roi,
        default=None,
        help=(
            "Batter back ROI for jersey OCR as 'x,y,w,h' fractions. "
            f"Default: {DEFAULT_CONFIG.batter_roi_rel}"
        ),
    )
    p.add_argument(
        "--pa-gap",
        type=float,
        default=DEFAULT_CONFIG.pa_max_gap_s,
        help="Seconds between pitches above which a new PA begins.",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for frame-accurate cuts (slower than -c copy).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = PipelineConfig()
    if args.pitch_roi is not None:
        cfg.pitch_roi_rel = args.pitch_roi
    if args.batter_roi is not None:
        cfg.batter_roi_rel = args.batter_roi
    cfg.pa_max_gap_s = args.pa_gap
    if args.reencode:
        cfg.use_stream_copy = False

    log = (lambda _msg: None) if args.quiet else (lambda msg: print(msg))

    try:
        analyze_inning(
            video_path=args.video,
            out_dir=args.out,
            inning=args.inning,
            config=cfg,
            progress=log,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
