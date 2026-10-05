"""rclone transport for the dataset and run directories (§8.2, Phase 6 §2 Q3).

``push`` moves *exactly* the files the dataset hash covers (``DATASET_HASH_PATTERNS`` through
``tree_files``) — so ``provenance/**`` goes up with the rest, and a remote copy can be checked by
re-computing the same hash. Survey sources never go: ``raw/`` and project roots are refused, and so
is a raw-suffixed file that ended up inside the dataset tree. ``pull`` brings a run directory down
whole except the trainer's scratch (``backend_out/``).

rclone's success is transport only. Integrity is decided by re-hashing what arrived: the pod
re-derives the dataset hash before training (``minegs.train.remote.worker``), and a RunPod run's
outputs are pulled through the manifest-verified path (``RunPodHandle.fetch_artifacts``), not here.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from minegs.core.errors import ContractError

#: The trainer's scratch tree; normalised outputs are already copied out of it.
RUN_EXCLUDE = ("backend_out/**",)


def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        raise ContractError("rclone not found on PATH (needed for RunPod sync)")
    return exe


def dataset_files(dataset_dir: Path) -> list[str]:
    """The dataset-relative files a push moves: those the dataset hash covers, raw refused."""
    from minegs.train.remote.bundle import dataset_file_table, require_dataset_dir

    return sorted(dataset_file_table(require_dataset_dir(dataset_dir)))


def push_command(
    dataset_dir: Path, remote: str, files_from: Path | None = None, transfers: int = 16
) -> list[str]:
    dataset_dir = Path(dataset_dir)
    dataset_files(dataset_dir)  # refusals (raw/, project root, raw-suffixed files) before argv
    return [
        "rclone",
        "copy",
        str(dataset_dir),
        remote,
        "--checksum",
        "--transfers",
        str(transfers),
        "--files-from",
        str(files_from) if files_from is not None else "<dataset hash file list>",
    ]


def pull_command(remote: str, local_dir: Path, transfers: int = 16) -> list[str]:
    argv = ["rclone", "copy", remote, str(local_dir), "--checksum", "--transfers", str(transfers)]
    for pat in RUN_EXCLUDE:
        argv += ["--exclude", pat]
    return argv


def push(dataset_dir: Path, remote: str, transfers: int = 16, dry_run: bool = False) -> list[str]:
    files = dataset_files(Path(dataset_dir))
    if dry_run:
        return push_command(dataset_dir, remote, None, transfers)
    exe = _rclone()
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "files.txt"
        lst.write_text("".join(f + "\n" for f in files))
        argv = push_command(dataset_dir, remote, lst, transfers)
        subprocess.run([exe, *argv[1:]], check=True)
    return argv


def pull(remote: str, local_dir: Path, transfers: int = 16, dry_run: bool = False) -> list[str]:
    argv = pull_command(remote, local_dir, transfers)
    if dry_run:
        return argv
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([_rclone(), *argv[1:]], check=True)
    return argv
