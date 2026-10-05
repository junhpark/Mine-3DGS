"""The network volume as the submitting machine reaches it (Phase 6 §2 Q2).

The pod mounts the volume and uses plain files. The submitter reaches the same volume through one
``sync.remote``: an rclone ``remote:path`` (RunPod's S3-compatible endpoint for the volume), or an
absolute local path where the same volume is mounted on this machine. Both stores expose the same
few operations; every integrity decision (hashes, manifests, write-once) is made above them, on the
bytes they move, so neither store is trusted to have moved the right ones.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from minegs.core.errors import ContractError
from minegs.train.remote.secrets import redact

_RCLONE_REMOTE = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]*$")


def _rel(rel: str) -> str:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not str(p) or str(p) == ".":
        raise ContractError(f"{rel!r} is not a path inside the volume")
    return str(p)


class RemoteStore:
    """Operations on the volume, by path relative to its root."""

    def describe(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def exists(self, rel: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def read_bytes(self, rel: str) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError

    def write_atomic(self, rel: str, data: bytes) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def upload_files(
        self, src_root: Path, files: list[str], dest_rel: str
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def download_tree(
        self, src_rel: str, dest: Path, exclude: Iterable[str] = ()
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LocalDirStore(RemoteStore):
    """The volume mounted at a local directory (and, in tests, a temp directory standing in)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or not self.root.is_dir():
            raise ContractError(f"storage root {root} is not an existing absolute directory")

    def describe(self) -> str:
        return f"local:{self.root}"

    def path(self, rel: str) -> Path:
        return self.root / _rel(rel)

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def read_bytes(self, rel: str) -> bytes | None:
        p = self.path(rel)
        return p.read_bytes() if p.is_file() else None

    def write_atomic(self, rel: str, data: bytes) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def upload_files(self, src_root: Path, files: list[str], dest_rel: str) -> None:
        dest = self.path(dest_rel)
        for rel in files:
            src = Path(src_root) / _rel(rel)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".partial")
            shutil.copyfile(src, tmp)
            os.replace(tmp, out)

    def download_tree(self, src_rel: str, dest: Path, exclude: Iterable[str] = ()) -> None:
        src = self.path(src_rel)
        if not src.is_dir():
            raise ContractError(f"{self.describe()}/{src_rel} does not exist")
        skip = tuple(str(PurePosixPath(e)).rstrip("/") + "/" for e in exclude)
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = f.relative_to(src).as_posix()
            if rel.startswith(skip):
                continue
            out = Path(dest) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)


class RcloneStore(RemoteStore):
    """The volume through an rclone remote. rclone's own success is transport, not integrity."""

    def __init__(self, remote: str) -> None:
        if not _RCLONE_REMOTE.match(remote):
            raise ContractError(f"sync.remote {remote!r} is not an rclone remote:path")
        self.remote = remote.rstrip("/")
        exe = shutil.which("rclone")
        if not exe:
            raise ContractError(
                "rclone is not on PATH; it is how this machine reaches the network volume "
                f"({remote}). Install rclone, or mount the volume and give its absolute path"
            )
        self.exe = exe

    def describe(self) -> str:
        return self.remote

    def _url(self, rel: str) -> str:
        return f"{self.remote}/{_rel(rel)}"

    def _run(self, *args: str, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
        out = subprocess.run([self.exe, *args], capture_output=True, check=False)
        if out.returncode not in ok:
            err = out.stderr.decode(errors="replace").strip()[-800:]
            raise ContractError(f"rclone {args[0]} failed ({out.returncode}): {redact(err)}")
        return out

    def exists(self, rel: str) -> bool:
        # rclone exits 3 (directory not found) or 4 (file not found) for a missing path.
        out = self._run("lsjson", "--stat", self._url(rel), ok=(0, 3, 4))
        return out.returncode == 0

    def read_bytes(self, rel: str) -> bytes | None:
        if not self.exists(rel):
            return None
        return self._run("cat", self._url(rel)).stdout

    def write_atomic(self, rel: str, data: bytes) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "blob"
            tmp.write_bytes(data)
            self._run("copyto", str(tmp), self._url(rel) + ".partial")
            self._run("moveto", self._url(rel) + ".partial", self._url(rel))

    def upload_files(self, src_root: Path, files: list[str], dest_rel: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "files.txt"
            lst.write_text("".join(_rel(f) + "\n" for f in files))
            self._run(
                "copy", "--checksum", "--files-from", str(lst), str(src_root), self._url(dest_rel)
            )

    def download_tree(self, src_rel: str, dest: Path, exclude: Iterable[str] = ()) -> None:
        args = ["copy", "--checksum", self._url(src_rel), str(dest)]
        for e in exclude:
            args += ["--exclude", f"{str(PurePosixPath(e)).rstrip('/')}/**"]
        self._run(*args)


def store_for(remote: str | None) -> RemoteStore:
    """The store a ``sync.remote`` names. Refused before anything external happens."""
    if not remote or not isinstance(remote, str):
        raise ContractError(
            "runner sync.remote is not set: it names how this machine reaches the pod's network "
            "volume (an rclone remote:path, or the absolute path where the volume is mounted)"
        )
    if remote.startswith("/"):
        return LocalDirStore(remote)
    return RcloneStore(remote)


__all__ = ["LocalDirStore", "RcloneStore", "RemoteStore", "store_for"]
