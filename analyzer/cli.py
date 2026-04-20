"""CLI entrypoint: analyze one or more GameChanger-style inning videos.

Two modes:

    # Single file
    python -m analyzer inning.mp4 --out results/inning_01 --inning T1

    # Batch: process every video in a folder, one subdir of --out-root per file
    python -m analyzer --batch convert_these/ --out-root results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import ANALYSIS_FILENAME, analyze_inning

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


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
    p.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Path to the inning video (mp4/mov). Required unless --batch is set.",
    )
    p.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Directory containing inning videos to process in bulk.",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output directory for a single-video run (required without --batch).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Parent output dir for --batch. One subfolder per input video.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --batch mode, skip inputs whose output dir already has analysis.json.",
    )
    p.add_argument(
        "--inning",
        type=str,
        default=None,
        help='Inning label for single-video runs (e.g. "T3"). '
        "Ignored in --batch mode — inning is inferred from the filename.",
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
        "--diff-threshold",
        type=int,
        default=DEFAULT_CONFIG.pitch_diff_threshold,
        help="Per-pixel intensity delta to count as motion. Lower = more sensitive.",
    )
    p.add_argument(
        "--peak-sigma",
        type=float,
        default=DEFAULT_CONFIG.pitch_peak_sigma,
        help="Adaptive threshold = median + sigma*MAD. Lower = more pitches detected.",
    )
    p.add_argument(
        "--min-pitch-gap",
        type=float,
        default=DEFAULT_CONFIG.pitch_min_gap_s,
        help="Minimum seconds between consecutive pitch events.",
    )
    p.add_argument(
        "--roll-window",
        type=float,
        default=DEFAULT_CONFIG.pitch_roll_window_s,
        help="Seconds of context used to compute the local pitch threshold. "
        "Bigger = smoother baseline; smaller = more responsive.",
    )
    p.add_argument(
        "--abs-floor",
        type=float,
        default=DEFAULT_CONFIG.pitch_abs_floor,
        help="Minimum absolute motion score required to count as a pitch, "
        "regardless of local baseline. Raise to suppress ambient noise.",
    )
    p.add_argument(
        "--debug-roi",
        type=Path,
        default=None,
        help="Write a PNG showing the pitch ROI overlaid on a mid-video frame, then exit.",
    )
    p.add_argument(
        "--debug-motion",
        type=Path,
        default=None,
        help="Also write the motion signal as CSV (columns: t_s,score) for plotting.",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for frame-accurate cuts (slower than -c copy).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.pitch_roi is not None:
        cfg.pitch_roi_rel = args.pitch_roi
    if args.batter_roi is not None:
        cfg.batter_roi_rel = args.batter_roi
    cfg.pa_max_gap_s = args.pa_gap
    cfg.pitch_diff_threshold = args.diff_threshold
    cfg.pitch_peak_sigma = args.peak_sigma
    cfg.pitch_min_gap_s = args.min_pitch_gap
    cfg.pitch_roll_window_s = args.roll_window
    cfg.pitch_abs_floor = args.abs_floor
    if args.reencode:
        cfg.use_stream_copy = False
    return cfg


def _discover_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"--batch path is not a directory: {folder}")
    videos = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    ]
    return videos


def _run_single(
    video: Path,
    out_dir: Path,
    inning: str | None,
    cfg: PipelineConfig,
    log,
    debug_motion_csv: Path | None = None,
) -> int:
    try:
        analyze_inning(
            video_path=video,
            out_dir=out_dir,
            inning=inning,
            config=cfg,
            progress=log,
            debug_motion_csv=debug_motion_csv,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    log = (lambda _msg: None) if args.quiet else (lambda msg: print(msg))

    if args.debug_roi is not None:
        if args.video is None:
            parser.error("--debug-roi needs a positional video argument")
        from .debug import save_roi_overlay

        save_roi_overlay(args.video, args.debug_roi, cfg)
        log(f"Wrote ROI overlay to {args.debug_roi}")
        return 0

    if args.batch is not None:
        if args.out_root is None:
            parser.error("--batch requires --out-root")
        try:
            videos = _discover_videos(args.batch)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if not videos:
            print(f"no videos found in {args.batch}", file=sys.stderr)
            return 2

        args.out_root.mkdir(parents=True, exist_ok=True)
        failures = 0
        for i, video in enumerate(videos, start=1):
            out_dir = args.out_root / video.stem
            if args.skip_existing and (out_dir / ANALYSIS_FILENAME).is_file():
                log(f"[{i}/{len(videos)}] skip {video.name} (already analyzed)")
                continue
            log(f"[{i}/{len(videos)}] === {video.name} → {out_dir} ===")
            rc = _run_single(
                video,
                out_dir,
                inning=video.stem,
                cfg=cfg,
                log=log,
                debug_motion_csv=args.debug_motion,
            )
            if rc != 0:
                failures += 1
        if failures:
            print(f"{failures} video(s) failed", file=sys.stderr)
            return 1
        return 0

    # Single-video mode
    if args.video is None:
        parser.error("positional 'video' is required unless --batch is set")
    if args.out is None:
        parser.error("--out is required in single-video mode")
    return _run_single(
        args.video, args.out, args.inning, cfg, log, debug_motion_csv=args.debug_motion
    )


if __name__ == "__main__":
    raise SystemExit(main())
