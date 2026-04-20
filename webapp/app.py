"""Local FastAPI review UI for inning analysis.

Run with:
    VIDEO_WORK_DIR=./results python -m webapp
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analyzer.config import DEFAULT_CONFIG
from analyzer.pipeline import (
    ANALYSIS_FILENAME,
    load_analysis,
    recut_from_markers,
    save_analysis,
)

WORK_DIR = Path(os.environ.get("VIDEO_WORK_DIR", "./results")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DIR = Path(__file__).parent

app = FastAPI(title="Inning Reviewer", version="0.1.0")

app.mount("/media", StaticFiles(directory=str(WORK_DIR)), name="media")
app.mount(
    "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
)
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))

POSITIONS = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
OUTCOMES = [
    "Single",
    "Double",
    "Triple",
    "Home Run",
    "Walk",
    "Hit by Pitch",
    "Strikeout (looking)",
    "Strikeout (swinging)",
    "Groundout",
    "Flyout",
    "Lineout",
    "Popout",
    "Foul Out",
    "Sacrifice Fly",
    "Sacrifice Bunt",
    "Fielder's Choice",
    "Reached on Error",
    "Other",
]


def _list_innings() -> list[dict]:
    items: list[dict] = []
    if not WORK_DIR.exists():
        return items
    for entry in sorted(WORK_DIR.iterdir()):
        analysis_path = entry / ANALYSIS_FILENAME
        if not analysis_path.is_file():
            continue
        try:
            data = load_analysis(entry)
        except Exception:
            continue
        pas = data.get("plate_appearances", [])
        reviewed = sum(1 for pa in pas if pa.get("edits", {}).get("reviewed"))
        items.append(
            {
                "name": entry.name,
                "inning": data.get("inning"),
                "pa_count": len(pas),
                "reviewed_count": reviewed,
            }
        )
    return items


def _load(name: str) -> tuple[Path, dict]:
    inning_dir = WORK_DIR / name
    if not (inning_dir / ANALYSIS_FILENAME).is_file():
        raise HTTPException(status_code=404, detail=f"No analysis for inning '{name}'")
    return inning_dir, load_analysis(inning_dir)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "innings": _list_innings(),
            "work_dir": str(WORK_DIR),
        },
    )


@app.get("/innings/{name}", response_class=HTMLResponse)
def inning_view(request: Request, name: str) -> HTMLResponse:
    _, analysis = _load(name)
    return templates.TemplateResponse(
        request,
        "inning.html",
        {
            "name": name,
            "analysis": analysis,
            "positions": POSITIONS,
            "outcomes": OUTCOMES,
        },
    )


@app.post("/innings/{name}/pa/{idx}")
async def update_pa(name: str, idx: int, request: Request) -> JSONResponse:
    inning_dir, analysis = _load(name)
    body = await request.json()
    edits_in = body.get("edits") or {}
    if not isinstance(edits_in, dict):
        raise HTTPException(status_code=400, detail="edits must be an object")

    allowed = {
        "jersey",
        "fielding_position",
        "out",
        "outcome",
        "notes",
        "reviewed",
        "pitch_count",
    }
    target = next(
        (pa for pa in analysis["plate_appearances"] if pa.get("index") == idx),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail=f"PA {idx} not found")

    edits = target.setdefault("edits", {})
    for k, v in edits_in.items():
        if k in allowed:
            edits[k] = v

    save_analysis(inning_dir, analysis)
    return JSONResponse({"ok": True, "pa": target})


@app.get("/innings/{name}/analysis.json")
def raw_analysis(name: str) -> JSONResponse:
    _, analysis = _load(name)
    return JSONResponse(analysis)


@app.get("/innings/{name}/mark", response_class=HTMLResponse)
def mark_view(request: Request, name: str) -> HTMLResponse:
    _, analysis = _load(name)
    full_filename = analysis.get("stored_video") or "full.mp4"
    existing_markers = [
        {"start_s": float(pa["start_s"])}
        for pa in analysis.get("plate_appearances", [])
    ]
    return templates.TemplateResponse(
        request,
        "mark.html",
        {
            "name": name,
            "analysis": analysis,
            "full_filename": full_filename,
            "existing_markers": existing_markers,
        },
    )


@app.post("/innings/{name}/markers")
async def post_markers(name: str, request: Request) -> JSONResponse:
    inning_dir, _ = _load(name)
    body = await request.json()
    markers = body.get("markers")
    if not isinstance(markers, list):
        raise HTTPException(status_code=400, detail="markers must be a list")
    try:
        result = recut_from_markers(inning_dir, markers, DEFAULT_CONFIG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse({"ok": True, "count": len(result["plate_appearances"])})


# --------------------------------------------------------------------------- #
# Jersey "combine" — gather all PAs for a given jersey across every inning.
# --------------------------------------------------------------------------- #

HIT_OUTCOMES = {"Single", "Double", "Triple", "Home Run"}
WALK_OUTCOMES = {"Walk", "Hit by Pitch"}
K_OUTCOMES = {"Strikeout (looking)", "Strikeout (swinging)"}
OUT_OUTCOMES = {
    "Groundout",
    "Flyout",
    "Lineout",
    "Popout",
    "Foul Out",
    "Sacrifice Fly",
    "Sacrifice Bunt",
    "Fielder's Choice",
}
COMBINES_DIR = "_combines"


def _resolve_jersey(pa: dict) -> str:
    edit = (pa.get("edits") or {}).get("jersey")
    if edit not in (None, ""):
        return str(edit).strip()
    auto = (pa.get("auto") or {}).get("jersey")
    if auto not in (None, ""):
        return str(auto).strip()
    return "?"


def _group_by_jersey() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in sorted(WORK_DIR.iterdir()):
        if entry.name == COMBINES_DIR:
            continue
        analysis_path = entry / ANALYSIS_FILENAME
        if not analysis_path.is_file():
            continue
        try:
            data = load_analysis(entry)
        except Exception:
            continue
        for pa in data.get("plate_appearances", []):
            jersey = _resolve_jersey(pa)
            groups[jersey].append({"inning": entry.name, "pa": pa})
    return dict(groups)


def _summary_for(entries: list[dict]) -> dict:
    pa = len(entries)
    hits = walks = strikeouts = outs = reviewed = 0
    pitch_sum = 0
    pitch_n = 0
    for e in entries:
        row = e["pa"]
        edits = row.get("edits") or {}
        outcome = edits.get("outcome")
        if outcome in HIT_OUTCOMES:
            hits += 1
        elif outcome in WALK_OUTCOMES:
            walks += 1
        elif outcome in K_OUTCOMES:
            strikeouts += 1
            outs += 1
        elif outcome in OUT_OUTCOMES:
            outs += 1
        elif edits.get("out") == "out":
            outs += 1
        if edits.get("reviewed"):
            reviewed += 1
        pc = edits.get("pitch_count")
        if pc is None:
            pc = row.get("pitch_count")
        if pc is not None:
            pitch_sum += int(pc)
            pitch_n += 1
    return {
        "pa": pa,
        "hits": hits,
        "walks": walks,
        "strikeouts": strikeouts,
        "outs": outs,
        "reviewed": reviewed,
        "avg_pitches": round(pitch_sum / pitch_n, 2) if pitch_n else None,
    }


def _jersey_dir_safe(jersey: str) -> str:
    safe = "".join(c for c in jersey if c.isalnum() or c in "-_")
    return safe or "unknown"


@app.get("/players", response_class=HTMLResponse)
def players_view(request: Request) -> HTMLResponse:
    groups = _group_by_jersey()
    rows: list[dict] = []
    for jersey in sorted(groups.keys(), key=lambda j: (j == "?", j)):
        entries = groups[jersey]
        combined = WORK_DIR / COMBINES_DIR / f"jersey_{_jersey_dir_safe(jersey)}.mp4"
        rows.append(
            {
                "jersey": jersey,
                "summary": _summary_for(entries),
                "combined_exists": combined.is_file(),
            }
        )
    return templates.TemplateResponse(
        request,
        "players.html",
        {"rows": rows, "work_dir": str(WORK_DIR)},
    )


@app.get("/players/{jersey}", response_class=HTMLResponse)
def player_view(request: Request, jersey: str) -> HTMLResponse:
    groups = _group_by_jersey()
    entries = groups.get(jersey, [])
    if not entries:
        raise HTTPException(status_code=404, detail=f"No PAs for jersey '{jersey}'")
    combined_rel = f"{COMBINES_DIR}/jersey_{_jersey_dir_safe(jersey)}.mp4"
    combined_path = WORK_DIR / combined_rel
    return templates.TemplateResponse(
        request,
        "player.html",
        {
            "jersey": jersey,
            "entries": entries,
            "summary": _summary_for(entries),
            "combined_rel": combined_rel if combined_path.is_file() else None,
        },
    )


@app.post("/players/{jersey}/combine")
def combine_clips(jersey: str) -> JSONResponse:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg not found on PATH")

    groups = _group_by_jersey()
    entries = groups.get(jersey)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No PAs for jersey '{jersey}'")

    combines = WORK_DIR / COMBINES_DIR
    combines.mkdir(parents=True, exist_ok=True)
    out_name = f"jersey_{_jersey_dir_safe(jersey)}.mp4"
    out_path = combines / out_name

    # Gather clip paths sorted by inning name then PA index for reel order.
    files: list[Path] = []
    for e in sorted(
        entries,
        key=lambda x: (x["inning"], int(x["pa"].get("index", 0))),
    ):
        clip = WORK_DIR / e["inning"] / e["pa"]["clip"]
        if clip.is_file():
            files.append(clip)
    if not files:
        raise HTTPException(status_code=404, detail="No clip files found on disk")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=str(combines)
    ) as list_file:
        for f in files:
            # Escape single quotes per ffmpeg concat spec.
            escaped = str(f.resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)

    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        list_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg concat failed: {proc.stderr.strip()}",
        )
    return JSONResponse(
        {
            "ok": True,
            "url": f"/media/{COMBINES_DIR}/{out_name}",
            "clip_count": len(files),
        }
    )
