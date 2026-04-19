"""Local FastAPI review UI for inning analysis.

Run with:
    VIDEO_WORK_DIR=./results python -m webapp
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analyzer.pipeline import ANALYSIS_FILENAME, load_analysis, save_analysis

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
        "home.html",
        {
            "request": request,
            "innings": _list_innings(),
            "work_dir": str(WORK_DIR),
        },
    )


@app.get("/innings/{name}", response_class=HTMLResponse)
def inning_view(request: Request, name: str) -> HTMLResponse:
    _, analysis = _load(name)
    return templates.TemplateResponse(
        "inning.html",
        {
            "request": request,
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
