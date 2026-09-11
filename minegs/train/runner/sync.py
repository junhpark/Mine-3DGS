"""rclone-based sync (§8.2). ``push`` moves *only* ``dataset/`` up; ``pull`` brings
``runs/<id>/`` (ply, log, run.json) down. ``raw/`` is never a valid source (§1.4)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from minegs.core.errors import ContractError

DATASET_INCLUDE = (
    "images/**",
    "sparse/**",
    "masks/**",
    "init_points.ply",
    "manifest.json",
    "centerline.csv",
)


def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        raise ContractError("rclone not found on PATH (needed for RunPod sync)")
    return exe


def push_command(dataset_dir: Path, remote: str, transfers: int = 16) -> list[str]:
    dataset_dir = Path(dataset_dir)
    if dataset_dir.name == "raw" or (
        (dataset_dir / "raw").exists() and not (dataset_dir / "manifest.json").exists()
    ):
        raise ContractError("refusing to push raw/ — only dataset/ goes to a pod (§1.4)")
    if not (dataset_dir / "manifest.json").exists():
        raise ContractError(f"{dataset_dir} is not a dataset (no manifest.json)")
    argv = ["rclone", "sync", str(dataset_dir), remote, "--transfers", str(transfers), "--checksum"]
    for pat in DATASET_INCLUDE:
        argv += ["--include", pat]
    return argv


def pull_command(
    remote: str,
    local_dir: Path,
    include: Sequence[str] = ("point_cloud/**", "log/**", "run.json", "stats/**"),
    transfers: int = 16,
) -> list[str]:
    argv = ["rclone", "copy", remote, str(local_dir), "--transfers", str(transfers)]
    for pat in include:
        argv += ["--include", pat]
    return argv


def push(dataset_dir: Path, remote: str, transfers: int = 16, dry_run: bool = False) -> list[str]:
    argv = push_command(dataset_dir, remote, transfers)
    if dry_run:
        return argv
    argv[0] = _rclone()
    subprocess.run(argv, check=True)
    return argv


def pull(
    remote: str,
    local_dir: Path,
    include: Sequence[str] | None = None,
    transfers: int = 16,
    dry_run: bool = False,
) -> list[str]:
    argv = pull_command(
        remote,
        local_dir,
        include or ("point_cloud/**", "log/**", "run.json", "stats/**"),
        transfers,
    )
    if dry_run:
        return argv
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    argv[0] = _rclone()
    subprocess.run(argv, check=True)
    return argv
