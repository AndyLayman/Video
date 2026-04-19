"""ffmpeg-based clip extraction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig


class FFmpegMissingError(RuntimeError):
    pass


def ensure_ffmpeg(config: PipelineConfig) -> str:
    path = shutil.which(config.ffmpeg_bin)
    if not path:
        raise FFmpegMissingError(
            f"ffmpeg binary '{config.ffmpeg_bin}' not found on PATH. "
            "Install ffmpeg (e.g. `apt install ffmpeg` or `brew install ffmpeg`)."
        )
    return path


def cut_clip(
    src: Path,
    dst: Path,
    start_s: float,
    end_s: float,
    config: PipelineConfig,
) -> Path:
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")
    ffmpeg = ensure_ffmpeg(config)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end_s - start_s

    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src)]
    if config.use_stream_copy:
        cmd += ["-t", f"{duration:.3f}", "-c", "copy", "-avoid_negative_ts", "1"]
    else:
        # Re-encode for frame-accurate cuts.
        cmd += [
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ]
    cmd.append(str(dst))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst.name}: {proc.stderr.strip()}")
    return dst
