"""Frame extraction with ffmpeg (command builder is pure; execution needs ffmpeg on PATH)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from minegs.core.errors import MissingDependencyError


def extract_command(
    video: str | Path,
    out_dir: str | Path,
    fps: float = 2.0,
    pattern: str = "v_%06d.png",
    scale_width: int | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> list[str]:
    vf = [f"fps={fps}"]
    if scale_width:
        vf.append(f"scale={scale_width}:-2")
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s is not None:
        argv += ["-ss", str(start_s)]
    argv += ["-i", str(video)]
    if duration_s is not None:
        argv += ["-t", str(duration_s)]
    argv += ["-vf", ",".join(vf), "-qscale:v", "2", str(Path(out_dir) / pattern)]
    return argv


def extract_frames(video: str | Path, out_dir: str | Path, **kw) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise MissingDependencyError(
            "ffmpeg", "video", "frame extraction (system package, not pip)"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(extract_command(video, out_dir, **kw), check=True)
    return sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.jpg"))


def probe_command(video: str | Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
