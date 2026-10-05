"""``RunInputBundle`` — everything one remote run reads, by identity (Phase 6 §5).

A remote run reads the materialised dataset (the exact files the dataset hash covers) plus the
sidecars it was bound to: a ``ChunkPlanRecord`` for a chunk run, a ``DepthSupervisionRecord`` for a
depth run. Neither sidecar is forced into the dataset hash; both are named here by digest, uploaded
to digest-addressed paths, and verified again in the pod. ``inputs.json`` is published after every
byte it names is in place, so a worker never starts from a half-uploaded set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel, canonical_json
from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file, sha256_tree_of, tree_files
from minegs.train.remote.layout import (
    CHUNK_PLAN_FILE,
    DATASET_CLAIM_FILE,
    INPUTS_FILE,
    RemoteLayout,
    safe_id,
)
from minegs.train.remote.store import RemoteStore

#: Never a dataset file, whatever directory it was put in (§5): survey sources stay local.
RAW_SUFFIXES = frozenset(
    {".e57", ".mp4", ".mov", ".avi", ".mkv", ".insv", ".insp", ".360", ".las", ".laz", ".lsproj"}
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkInput(_Strict):
    chunk_id: str
    plan_id: str
    plan_digest: str
    #: Pod-side path of the uploaded plan. A location, not an identity: the worker checks the
    #: digest and re-verifies the plan against the dataset it actually reads.
    path: str


class DepthInput(_Strict):
    supervision_id: str
    artifact_sha256: str
    path: str


class RunInputBundle(VersionedModel):
    """``jobs/<run_id>/inputs.json`` (docs/PHASE6_CONTRACT.md §5)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    run_id: str
    dataset_id: str
    dataset_hash: str
    dataset_files: int
    profile: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    backend: str
    chunk: ChunkInput | None = None
    depth_supervision: DepthInput | None = None
    image: str
    requested_gpu_types: list[str]
    gpu_count: int = 1
    cuda_archs: list[str]
    network_volume_id: str
    volume_mount: str
    #: Pod-side absolute paths.
    paths: dict[str, str]
    submitter: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    bundle_digest: str = ""

    def content(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d.pop("bundle_digest", None)
        return d

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.content()).encode()).hexdigest()

    def sealed(self) -> RunInputBundle:
        return self.model_copy(update={"bundle_digest": self.digest()})

    def check_digest(self) -> None:
        if not self.bundle_digest or self.bundle_digest != self.digest():
            raise ContractError(
                f"inputs.json for run {self.run_id} does not hash to its bundle_digest; the input "
                "set was edited after it was published"
            )


# ---------------------------------------------------------------- the dataset that goes up


def require_dataset_dir(path: str | Path) -> Path:
    """A materialised dataset directory, and nothing that holds survey sources (§5).

    The directory itself must not be a symlink either: resolving it first would hide the link,
    and every check below would then describe a directory nobody named.
    """
    if Path(path).is_symlink():
        raise ContractError(
            f"{path} is a symlink; give the dataset directory itself (nothing is uploaded "
            "through a link)"
        )
    p = Path(path).resolve()
    if p.name == "raw" or "raw" in p.parts[-2:-1]:
        raise ContractError(f"{path}: raw/ never leaves this machine; give the dataset/ directory")
    if not (p / "manifest.json").is_file():
        hint = " (a project root? give its dataset/ directory)" if (p / "raw").exists() else ""
        raise ContractError(f"{path} is not a dataset: no manifest.json{hint}")
    return p


def _refuse_link(root: Path, f: Path, what: str) -> Path:
    """``f`` as a plain file under ``root``: no symlinked component, nothing resolving outside.

    A link is never followed into a pod (§5): it can point at survey sources or any other file
    on this machine, and the bytes uploaded would then not be the dataset's own.
    """
    rel = f.relative_to(root)
    cur = root
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ContractError(
                f"{what} {root}: {cur.relative_to(root).as_posix()} is a symlink; nothing is "
                "uploaded through a link (it can point at raw data or any file on this machine). "
                "Copy the file into place"
            )
    real = f.resolve()
    if not real.is_relative_to(root.resolve()):
        raise ContractError(f"{what} {root}: {rel.as_posix()} resolves outside it ({real})")
    if f.suffix.lower() in RAW_SUFFIXES or real.suffix.lower() in RAW_SUFFIXES:
        raise ContractError(
            f"{what} {root} holds survey source files ({rel.as_posix()}); raw data never goes "
            "to a pod. Remove it from the dataset (it belongs in raw/)"
        )
    return f


def dataset_file_table(dataset_dir: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of exactly the files the dataset hash covers.

    Raw sources are refused, and so is any symlink on the way to a file (§5).
    """
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    files = [
        _refuse_link(dataset_dir, f, "dataset")
        for f in tree_files(dataset_dir, DATASET_HASH_PATTERNS)
    ]
    return {f.relative_to(dataset_dir).as_posix(): sha256_file(f) for f in files}


def dataset_hash_of(table: dict[str, str]) -> str:
    """The dataset hash of a file table — ``sha256_tree`` of the bytes that were read."""
    return sha256_tree_of(table)


# ---------------------------------------------------------------- publishing


def _json(model: BaseModel) -> bytes:
    return model.model_dump_json(indent=2).encode()


def publish_inputs(
    store: RemoteStore,
    layout: RemoteLayout,
    bundle: RunInputBundle,
    dataset_dir: Path,
    table: dict[str, str],
    *,
    chunk_plan_file: Path | None = None,
    depth_dir: Path | None = None,
    depth_files: list[str] | None = None,
) -> RunInputBundle:
    """Upload the dataset and sidecars, then publish ``inputs.json``. Returns the sealed bundle.

    Write-once and collision checks come first, before any byte moves: a run id that already has
    a job or a run on the volume is refused, and so is a content-addressed dataset path whose
    claim names another hash. A claimed dataset path is reused as it is, never re-uploaded.
    """
    run_id = safe_id(bundle.run_id, "run_id")
    for rel in (layout.job(run_id), layout.run(run_id)):
        if store.exists(rel):
            raise ContractError(
                f"{store.describe()}/{rel} already exists: a remote run directory is written once, "
                "and a new run must not inherit another's job or outputs"
            )
    ds_rel = layout.dataset(bundle.dataset_hash)
    claim_rel = f"{ds_rel}/{DATASET_CLAIM_FILE}"
    claim = {"dataset_id": bundle.dataset_id, "dataset_hash": bundle.dataset_hash}
    raw_claim = store.read_bytes(claim_rel)
    if raw_claim is not None:
        import json

        try:
            have = json.loads(raw_claim)
        except ValueError as e:
            raise ContractError(f"{claim_rel} is not JSON ({e}); refusing to write over it") from e
        if have != claim:
            raise ContractError(
                f"{store.describe()}/{claim_rel} claims {have}, not {claim}: the dataset path for "
                "this hash holds something else. Nothing was overwritten"
            )
    if raw_claim is None:
        # Content-addressed and written once: the claim goes last, so a claimed path is a whole
        # upload, and a claimed path is never written again (the pod re-hashes what it reads).
        store.upload_files(dataset_dir, sorted(table), ds_rel)
        store.write_atomic(claim_rel, canonical_json(claim).encode())

    if bundle.chunk is not None:
        if chunk_plan_file is None:
            raise ContractError("a chunk run needs its plan file to upload")
        # Under the name the bundle gives the pod, whatever the local file was called.
        store.write_atomic(
            f"{layout.chunk_plan(bundle.chunk.plan_digest)}/{CHUNK_PLAN_FILE}",
            Path(chunk_plan_file).read_bytes(),
        )
    if bundle.depth_supervision is not None:
        if depth_dir is None or not depth_files:
            raise ContractError(
                "a depth-supervised run needs its verified artifact files to upload"
            )
        # Exactly the files the verified record names, nothing re-enumerated after the check.
        for rel in depth_files:
            _refuse_link(depth_dir, depth_dir / rel, "depth artifact")
        store.upload_files(
            depth_dir, sorted(depth_files), layout.depth(bundle.depth_supervision.artifact_sha256)
        )

    sealed = bundle.sealed()
    store.write_atomic(f"{layout.job(run_id)}/{INPUTS_FILE}", _json(sealed))
    return sealed


def load_bundle(path: str | Path) -> RunInputBundle:
    p = Path(path)
    if p.name != INPUTS_FILE or not p.is_file():
        raise ContractError(
            f"{path}: no published {INPUTS_FILE}; a worker does not start from a partial input set"
        )
    bundle = RunInputBundle.load(p)
    bundle.check_digest()
    return bundle


__all__ = [
    "RAW_SUFFIXES",
    "ChunkInput",
    "DepthInput",
    "RunInputBundle",
    "dataset_file_table",
    "dataset_hash_of",
    "load_bundle",
    "publish_inputs",
    "require_dataset_dir",
]
