#!/usr/bin/env python3
"""Launch video_ball_detector under debugpy and announce readiness."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

try:
    import debugpy  # type: ignore
except ImportError as exc:  # pragma: no cover - debugpy must be installed in venv
    raise SystemExit(f"debugpy import failed: {exc}")

DEFAULT_TARGET = Path(__file__).with_name("video_ball_detector.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Python target under debugpy")
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Python file to execute once a debugger attaches",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DEBUGPY_HOST", "0.0.0.0"),
        help="Address debugpy listens on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DEBUGPY_PORT", "5678")),
        help="Port debugpy listens on",
    )
    parsed = parser.parse_args()

    listen_host = parsed.host
    listen_port = parsed.port
    debugpy.listen((listen_host, listen_port))
    print(f"[debugpy] listening on {listen_host}:{listen_port}", flush=True)
    debugpy.wait_for_client()

    target_path = Path(parsed.target).resolve()
    if not target_path.exists():
        raise SystemExit(f"Target script not found: {target_path}")

    sys.argv = [str(target_path), *parsed.args]
    runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
    main()
