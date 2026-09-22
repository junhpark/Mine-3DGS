"""Provenance (§9): every artifact records input hashes, config hash, git SHA, tool versions,
parent IDs. ``raw -> dataset -> run -> eval -> export``.

IDs: ``<prefix>_<YYYYMMDD>_<hex6>``  e.g. ``gsplat_20260910_a91f2c``.
"""

from __future__ import annotations

import hashlib
import importlib
import platform
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import minegs
from minegs.core.config import config_hash


class SourceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str
    size_bytes: int | None = None


class ProvenanceRecord(BaseModel):
    """Attached to dataset manifest, run.json, eval.json, export.json."""

    model_config = ConfigDict(extra="forbid")

    minegs_version: str = Field(default_factory=lambda: minegs.__version__)
    git_commit: str = Field(default="unknown")
    config_hash: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_ids: list[str] = Field(default_factory=list)
    source_assets: list[SourceAsset] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    host: str = Field(default_factory=platform.node)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_tree(root: str | Path, patterns: tuple[str, ...] = ("**/*",)) -> str:
    """Order-independent hash of a directory (relative path + content hash per file)."""
    root = Path(root)
    files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def source_asset(path: str | Path, rel_to: str | Path | None = None) -> SourceAsset:
    p = Path(path)
    rel = str(p.relative_to(rel_to)) if rel_to else str(p)
    return SourceAsset(path=rel, sha256=sha256_file(p), size_bytes=p.stat().st_size)


def git_commit(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root else Path(minegs.__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _package_version(name: str) -> str | None:
    """Version of an installed package.

    Prefer installed metadata: some packages (pye57) bind ``__version__`` to a *module*, and
    ``str()`` on that leaks a local filesystem path into the provenance record.
    """
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    mod = importlib.import_module(name)
    v = getattr(mod, "__version__", None)
    if isinstance(v, str):
        return v
    inner = getattr(v, "__version__", None)  # module-shaped __version__
    return inner if isinstance(inner, str) else None


_TOOLS = (
    "numpy",
    "scipy",
    "pydantic",
    "torch",
    "gsplat",
    "pycolmap",
    "pye57",
    "pdal",
    "viser",
    "open3d",
)


def tool_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version(), "minegs": minegs.__version__}
    for name in _TOOLS:
        try:
            v = _package_version(name)
        except Exception:  # absent optional dependency is the normal case
            continue
        if v is not None:
            out[name] = v
    for cmd in ("colmap", "ffmpeg", "pdal", "rclone"):
        v = _cli_version(cmd)
        if v:
            out[cmd] = v
    if extra:
        out.update(extra)
    return out


def _cli_version(cmd: str) -> str | None:
    """The tool's version, or ``None`` when it could not be established.

    This used to run the probe with ``check=False`` and record ``stdout or stderr`` whatever
    happened, which meant a *failed* probe was written down as the version. COLMAP 3.9.1 does
    not accept ``--version``, so every provenance record it touched carried

        "E... colmap.cc:158] Command `--version` not recognize"

    in the slot where the version of the engine that produced a reconstruction belongs, and
    nothing said so. A probe that exits non-zero establishes nothing, and nothing is ``None``.

    ``-h`` is tried second because that is where COLMAP prints its banner; it runs only after
    ``--version`` has already failed, so a tool that answers the first question is never asked
    the second.
    """
    for argv in ([cmd, "--version"], [cmd, "-h"]):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            continue
        line = next(
            (ln.strip() for ln in (out.stdout or out.stderr).splitlines() if ln.strip()), ""
        )
        if line:
            return line[:80]
    return None


def make_id(prefix: str, when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"{prefix}_{when.strftime('%Y%m%d')}_{secrets.token_hex(3)}"


def stamp(
    config: Any = None,
    *,
    parents: list[str] | None = None,
    assets: list[SourceAsset] | None = None,
    tools: dict[str, str] | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        git_commit=git_commit(),
        config_hash=config_hash(config) if config is not None else "",
        parent_ids=list(parents or []),
        source_assets=list(assets or []),
        tool_versions=tool_versions(tools),
    )
