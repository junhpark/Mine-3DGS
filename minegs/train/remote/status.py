"""Durable job status and the output manifest (Phase 6 §6, §9).

The provider says whether a pod exists, not whether training succeeded: RunPod's pod query has
no container exit code, and a pod that is EXITED, TERMINATED or gone tells nothing about the
trainer. So the worker writes what happened to the network volume itself — ``status.json`` with
the exit codes, and ``output_manifest.json`` with a digest per output file — and success is
decided from those, re-checked on this side, never from the pod's lifecycle.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file

#: Not an output: the trainer's scratch tree, whose normalised contents are copied out of it.
OUTPUT_EXCLUDE = ("backend_out",)


class JobState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


FINAL_STATES = (JobState.SUCCEEDED, JobState.FAILED)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStatus(VersionedModel):
    """``jobs/<run_id>/status.json`` — written only by the worker, atomically."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    run_id: str
    state: JobState
    started_at: str | None = None
    completed_at: str | None = None
    #: The worker's own exit code: 0 only when the run succeeded and its outputs were recorded.
    exit_code: int | None = None
    #: The trainer process's exit code, as the worker's process handle saw it.
    trainer_exit_code: int | None = None
    failure_stage: str | None = None
    message: str | None = None
    input_bundle_digest: str | None = None
    pod_dataset_hash: str | None = None
    run_record_path: str | None = None
    output_manifest_path: str | None = None
    output_manifest_sha256: str | None = None

    @property
    def final(self) -> bool:
        return self.state in FINAL_STATES

    @property
    def succeeded(self) -> bool:
        return (
            self.state is JobState.SUCCEEDED
            and self.exit_code == 0
            and self.trainer_exit_code == 0
            and self.output_manifest_sha256 is not None
        )


def write_json_atomic(path: Path, model: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(model.model_dump_json(indent=2))
    os.replace(tmp, path)
    return path


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputEntry(_Strict):
    path: str
    size: int
    sha256: str


class RemoteOutputRecord(VersionedModel):
    """``runs/<run_id>/output_manifest.json``: every output file, by size and digest."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    run_id: str
    status: str
    exit_code: int | None = None
    dataset_hash: str | None = None
    image: str | None = None
    run_json_sha256: str | None = None
    entries: list[OutputEntry] = Field(default_factory=list)
    output_tree_digest: str
    created_at: str


def tree_digest(entries: list[OutputEntry]) -> str:
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e.path):
        h.update(f"{e.path}\0{e.size}\0{e.sha256}\n".encode())
    return h.hexdigest()


def _output_files(run_dir: Path, manifest_name: str, strict: bool = False) -> list[Path]:
    """The files a run's outputs are. ``strict`` (the pulling side) leaves nothing out: a
    ``.partial`` the pod skipped is, once pulled, a file the manifest does not name."""
    out = []
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir)
        if rel.parts and rel.parts[0] in OUTPUT_EXCLUDE:
            continue
        if not p.is_file() or rel.as_posix() == manifest_name:
            continue
        if strict or not p.name.endswith(".partial"):
            out.append(p)
    return out


def output_entries(run_dir: Path, manifest_name: str, strict: bool = False) -> list[OutputEntry]:
    return [
        OutputEntry(
            path=p.relative_to(run_dir).as_posix(), size=p.stat().st_size, sha256=sha256_file(p)
        )
        for p in _output_files(run_dir, manifest_name, strict)
    ]


def verify_output_tree(root: Path, manifest: RemoteOutputRecord, manifest_name: str) -> str:
    """Every manifest entry present with its size and digest, and nothing else. Returns the digest
    re-computed from the bytes on this side."""
    have = {e.path: e for e in output_entries(root, manifest_name, strict=True)}
    want = {e.path: e for e in manifest.entries}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    if missing or extra:
        raise ContractError(
            f"run {manifest.run_id}: the pulled outputs are not the manifest's — missing "
            f"{missing[:5]}, not in the manifest {extra[:5]}"
        )
    bad = [p for p, e in want.items() if (have[p].size, have[p].sha256) != (e.size, e.sha256)]
    if bad:
        raise ContractError(
            f"run {manifest.run_id}: {len(bad)} pulled file(s) differ from the manifest "
            f"({bad[:5]}); the outputs changed between the pod and here"
        )
    got = tree_digest(list(have.values()))
    if got != manifest.output_tree_digest:
        raise ContractError(
            f"run {manifest.run_id}: the manifest's output_tree_digest does not match its own "
            "entries; the manifest was edited"
        )
    return got


__all__ = [
    "FINAL_STATES",
    "OUTPUT_EXCLUDE",
    "JobState",
    "JobStatus",
    "OutputEntry",
    "RemoteOutputRecord",
    "output_entries",
    "tree_digest",
    "verify_output_tree",
    "write_json_atomic",
]
