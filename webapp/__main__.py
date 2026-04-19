"""Run the review UI: `python -m webapp` (or `python -m webapp --port 8765`)."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="webapp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory containing inning subfolders. "
        "Defaults to $VIDEO_WORK_DIR or ./results.",
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.work_dir:
        os.environ["VIDEO_WORK_DIR"] = args.work_dir

    uvicorn.run(
        "webapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
