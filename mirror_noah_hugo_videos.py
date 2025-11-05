"""
Mirror (horizontally flip) the noah_hugo_<n>.mp4 videos.

Creates new files alongside the originals with suffix `_mirrored`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def mirror_videos(base_dir: Path, indices: Iterable[int]) -> None:
    """Flip the specified videos horizontally using ffmpeg."""
    for idx in indices:
        src = base_dir / f"noah_hugo_{idx}.mp4"
        if not src.exists():
            print(f"Skipping {src.name}: file not found.")
            continue

        dst = src.with_name(f"{src.stem}_mirrored{src.suffix}")

        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file if it exists
            "-i",
            str(src),
            "-vf",
            "hflip",
            "-fps_mode",
            "passthrough",  # avoid duplicating frames at extreme fps
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            str(dst),
        ]

        print(f"Mirroring {src.name} -> {dst.name}")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it is on PATH."
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed for {src.name}") from exc


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    mirror_videos(project_dir, range(1, 21))
