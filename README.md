# GameChanger Inning Video Analyzer

Local pipeline that takes a fixed-backstop GameChanger-style **inning video**
(~20 minutes), detects pitch events, segments the video into per-batter clips,
takes a best-effort guess at jersey number / fielding position / outcome, and
opens a small local web UI for you to review and correct the calls.

This is intentionally a **fully local** v1. There is no cloud upload, no API
calls, and no model training. The CV is heuristic — your manual review is the
source of truth. A future request will wire the corrected `analysis.json` into
your stats app.

## What you get per inning

```
results/inning_01/
├── full.mp4                  # the original (ready to push to YouTube)
├── analysis.json             # auto + edited fields
└── clips/
    ├── pa_01.mp4
    ├── pa_02.mp4
    └── ...
```

`analysis.json` shape (per plate appearance):
```json
{
  "index": 1,
  "clip": "clips/pa_01.mp4",
  "start_s": 4.0, "end_s": 58.0,
  "pitch_count": 3,
  "auto":  { "jersey": "12", "in_play": true, "likely_position": "SS", ... },
  "edits": { "jersey": null, "fielding_position": null, "out": null,
             "outcome": null, "notes": null, "reviewed": false }
}
```

## Install

```bash
pip install -r requirements.txt
# ffmpeg is required at runtime:
sudo apt install ffmpeg          # Linux
brew install ffmpeg              # macOS
```

EasyOCR (optional, used for jersey numbers) downloads its model on first use.
If you skip it, `auto.jersey` will simply be `null` and you fill it in by hand.

## Run the analyzer

**Single inning:**
```bash
python -m analyzer path/to/inning_top1.mp4 \
  --out results/inning_01 \
  --inning T1
```

**Batch mode** — point it at a folder of inning videos and it processes each
one into its own subfolder of `--out-root` (named after the input file):
```bash
python -m analyzer --batch convert_these/ --out-root results/
# add --skip-existing to resume without re-processing finished videos
```

Useful flags:

| Flag | What it does |
|---|---|
| `--pa-gap 35` | Seconds between pitches before declaring a new batter. Lower if you see batters merged together; raise if a single batter gets split. |
| `--pitch-roi x,y,w,h` | Override the pitcher→plate motion ROI. Fractions of frame, e.g. `0.33,0.05,0.34,0.6`. |
| `--batter-roi x,y,w,h` | Override the batter-back ROI used for jersey OCR. |
| `--reencode` | Frame-accurate clip cuts (slower; default is fast `-c copy` which snaps to keyframes). |
| `--quiet` | Suppress per-PA progress logs. |

## Review the calls

```bash
python -m webapp --work-dir results
# opens http://127.0.0.1:8765
```

The home page lists every inning under `results/`. Each inning page plays
each PA's clip inline and gives you a form: jersey #, pitch count, outcome,
fielding position, out/safe, notes, and a "reviewed" toggle. Edits save back
to `analysis.json` immediately.

## What the auto-analysis can and can't do

| Field | How it's computed | Realistic accuracy |
|---|---|---|
| `pitch_count` | Motion peaks in the pitcher→plate corridor | ~80–90% |
| PA boundary | Inter-pitch gap > `--pa-gap` | High when gaps are clear |
| `jersey` | EasyOCR on the batter ROI between pitches | 30–60% — manual review essential |
| `in_play` | Sustained motion after the last pitch | Decent |
| `likely_position` | Frame-region of post-contact motion | Rough — a starting guess |
| `out` | Not auto-detected | Always your call |
| `outcome` | Not auto-detected | Always your call |

## Layout

```
analyzer/
  config.py         # tunable defaults
  video_io.py       # cv2 metadata + ROI helpers (cv2 lazy-imported)
  motion.py         # frame-difference motion signal extraction
  pitch_detect.py   # peak-finding on the motion signal
  pa_segment.py     # group pitches into plate appearances
  outcome.py        # best-effort post-pitch outcome guess
  jersey_ocr.py     # EasyOCR on batter ROI
  clip.py           # ffmpeg clip cutting
  pipeline.py       # end-to-end orchestration
  cli.py            # argparse entrypoint
webapp/
  app.py            # FastAPI review UI
  templates/        # Jinja2 templates
  static/style.css
```

## Limits to be aware of

- Heuristic CV only — designed for the **fixed backstop** camera angle. A
  moving camera will break pitch detection.
- "Single to **third baseman**" specificity isn't fully automatable here; we
  give you `likely_position` as a starting guess and your edit is the truth.
- `-c copy` clip cuts snap to keyframes and may start a frame or two early or
  late. Pass `--reencode` for frame-accurate cuts (slower).
