"""``ChunkPlanRecord`` — one long-tunnel dataset cut into training chunks (Phase 5 AD-1…AD-3).

A plan is a derived execution artifact, not part of the dataset. It lives at
``<dataset>/chunks/<plan_id>/chunk_plan.json``, outside ``DATASET_HASH_PATTERNS``, and records the
dataset id and hash and the centerline sha256 it was made against. Writing a plan therefore never
changes dataset identity, and a dataset or axis that changed after the plan was made refuses it.

Each chunk has a **core**, the chainage it owns in every final result, and a **support**, the core
widened by the overlap, which is training context only. Cores tile the axis with one ownership
rule (``owner_index``): ``[b_i, b_{i+1})``, the last one closed at the axis end. Supports overlap;
cores never do.

Images are chosen by capture group, never one by one: a group whose chainage span meets a
chunk's support comes in whole (a 360 ring or a video segment is never cut), and only after the
global split has been applied (``manifest.train_images()`` minus ``test_images()``). A group that
straddles a boundary therefore trains in both chunks, and the image support a chunk actually has
(``actual_image_support_m``) can be wider than its nominal support; that is recorded, not hidden.

The plan is a pure function of (dataset, centerline, policy). ``plan_digest`` hashes everything but
provenance, ``plan_id`` is derived from it, and ``verify_chunk_plan`` re-plans and compares.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minegs.core.config import VersionedModel, canonical_json
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest, spans_overlap
from minegs.core.provenance import ProvenanceRecord, sha256_file, stamp

PLAN_DIR = "chunks"
PLAN_FILE = "chunk_plan.json"
OWNERSHIP_RULE = "core_half_open_last_closed"
#: Chainage tolerance of the ownership rule and of the plan comparisons, in metres.
EPS_M = 1e-9


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkPolicy(_Strict):
    core_length_m: float
    overlap_m: float
    ownership_rule: str = OWNERSHIP_RULE


class PlannedChunk(_Strict):
    chunk_id: str
    ordinal: int
    core_range_m: tuple[float, float]
    support_range_m: tuple[float, float]
    #: Train capture groups whose span meets the support, whole.
    capture_groups: list[str]
    #: Their members that the global split lets train, sorted.
    images: list[str]
    #: Outer chainage of the selected groups' spans; may exceed the support.
    actual_image_support_m: tuple[float, float]
    #: Every group (any split) whose span meets the support: the views a chunk model is rendered
    #: from when its surface is built (Phase 5 §7). Rendering is not training.
    view_groups: list[str]
    views: list[str]


class ChunkPlanRecord(VersionedModel):
    """``chunk_plan.json`` (docs/PHASE5_CONTRACT.md §4)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    plan_id: str = Field(min_length=1)
    plan_digest: str = Field(min_length=64, max_length=64)
    dataset_id: str
    dataset_hash: str
    centerline_file: str
    centerline_sha256: str
    axis_range_m: tuple[float, float]
    policy: ChunkPolicy
    chunks: list[PlannedChunk]
    #: Non-training groups with no chainage span: in no chunk, rendered for none.
    unplaced_groups: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    def content(self) -> dict[str, Any]:
        """Everything the digest covers (all but identity and provenance)."""
        return _content_of(self.model_dump(mode="json"))

    def boundaries(self) -> np.ndarray:
        """The interior core boundaries ``b_1 … b_{n-1}``."""
        return np.array([c.core_range_m[0] for c in self.chunks[1:]], dtype=np.float64)

    def owner_index(self, s: Any) -> np.ndarray:
        """Index of the one chunk that owns each chainage (the only ownership function)."""
        return owner_index(self.boundaries(), s)

    def chunk(self, chunk_id: str) -> PlannedChunk:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        raise ContractError(
            f"chunk {chunk_id!r} is not in plan {self.plan_id} "
            f"({[c.chunk_id for c in self.chunks]})"
        )

    def core_extent(self) -> tuple[float, float]:
        return (self.chunks[0].core_range_m[0], self.chunks[-1].core_range_m[1])


def owner_index(boundaries: np.ndarray, s: Any) -> np.ndarray:
    """``[b_i, b_{i+1})``, last closed: a chainage on a boundary belongs to the chunk after it."""
    s = np.asarray(s, dtype=np.float64)
    return np.searchsorted(np.asarray(boundaries, dtype=np.float64), s + EPS_M, side="right")


def _content_of(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in ("plan_id", "plan_digest", "provenance")}


def _digest(content: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(content).encode()).hexdigest()


def _refuse(why: str) -> ContractError:
    return ContractError(f"chunk plan: {why}")


# ---------------------------------------------------------------- planning


def check_policy(core_length_m: float, overlap_m: float) -> None:
    if not (math.isfinite(core_length_m) and core_length_m > 0):
        raise _refuse(f"core_length_m must be a positive length, got {core_length_m}")
    if not (math.isfinite(overlap_m) and overlap_m >= 0):
        raise _refuse(f"overlap_m must be >= 0, got {overlap_m}")
    if not overlap_m < core_length_m:
        raise _refuse(
            f"overlap_m {overlap_m} must be smaller than core_length_m {core_length_m}; "
            "otherwise a chunk's context would reach past its neighbour's core"
        )


def core_ranges(
    s_start: float, s_end: float, core_length_m: float, overlap_m: float
) -> list[tuple[float, float]]:
    """Cores of length L from the axis start; the last one takes the remainder.

    A remainder no longer than the overlap is merged into the core before it: the previous
    chunk's support already reaches the axis end, so a chunk of its own would train on nothing
    its neighbour does not, and its core could be a sliver with no image at all.
    """
    if not s_end - s_start > EPS_M:
        raise _refuse(f"the axis spans [{s_start}, {s_end}]; there is nothing to cut")
    n = max(1, math.ceil(round((s_end - s_start) / core_length_m, 9)))
    b = [s_start + k * core_length_m for k in range(n)] + [s_end]
    if n > 1 and s_end - b[n - 1] <= overlap_m + EPS_M:
        del b[n - 1]
    return [(float(b[i]), float(b[i + 1])) for i in range(len(b) - 1)]


def _axis(dataset_dir: Path, manifest: Manifest):
    from minegs.train.supervision.support import dataset_centerline

    if manifest.centerline is None:
        raise _refuse(
            f"{manifest.dataset_id} declares no centerline; chunks are cut along chainage and "
            "there is no chainage without an axis"
        )
    cl = dataset_centerline(dataset_dir, manifest)
    assert cl is not None
    return cl


def plan_content(
    dataset_dir: str | Path,
    core_length_m: float,
    overlap_m: float,
    *,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """The plan for a dataset and a policy, as plain data (deterministic; no provenance)."""
    from minegs.core.provenance import sha256_tree
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    ds = Path(dataset_dir)
    manifest = manifest or Manifest.load_dataset(ds)
    check_policy(float(core_length_m), float(overlap_m))
    cl = _axis(ds, manifest)
    s0, s1 = float(cl.s_start), float(cl.s_end)

    train_imgs = set(manifest.train_images()) - set(manifest.test_images())
    train_groups = set(manifest.split.train_groups)
    spans: dict[str, tuple[float, float]] = {}
    unplaced: list[str] = []
    for gid in sorted(manifest.capture_groups):
        g = manifest.capture_groups[gid]
        span = g.span()
        trains = gid in train_groups and any(m in train_imgs for m in g.members)
        if span is None:
            if trains:
                raise _refuse(
                    f"train group {gid} has no chainage span, so no chunk can be chosen for it; "
                    "chunking would silently drop its images from training"
                )
            unplaced.append(gid)
            continue
        spans[gid] = (float(span[0]), float(span[1]))

    chunks: list[dict[str, Any]] = []
    covered: set[str] = set()
    cores = core_ranges(s0, s1, float(core_length_m), float(overlap_m))
    for i, (lo, hi) in enumerate(cores):
        sup = (max(s0, lo - float(overlap_m)), min(s1, hi + float(overlap_m)))
        view_groups = sorted(g for g, sp in spans.items() if spans_overlap(sp, [sup]))
        groups = [
            g
            for g in view_groups
            if g in train_groups
            and any(m in train_imgs for m in manifest.capture_groups[g].members)
        ]
        images = sorted(
            m for g in groups for m in manifest.capture_groups[g].members if m in train_imgs
        )
        cid = f"K{i:03d}"
        if not images:
            raise _refuse(
                f"chunk {cid} (core {lo:g}-{hi:g} m, support {sup[0]:g}-{sup[1]:g} m) has no "
                "training image after the global split; choose another core length or overlap"
            )
        covered.update(images)
        chunks.append(
            {
                "chunk_id": cid,
                "ordinal": i,
                "core_range_m": [lo, hi],
                "support_range_m": [sup[0], sup[1]],
                "capture_groups": groups,
                "images": images,
                "actual_image_support_m": [
                    min(spans[g][0] for g in groups),
                    max(spans[g][1] for g in groups),
                ],
                "view_groups": view_groups,
                "views": sorted(m for g in view_groups for m in manifest.capture_groups[g].members),
            }
        )
    orphans = sorted(train_imgs - covered)
    if orphans:
        raise _refuse(
            f"{len(orphans)} training image(s) fall in no chunk (e.g. {orphans[:3]}); a chunked "
            "run set would train on less than the dataset does"
        )
    cl_file = ds / manifest.centerline.file  # type: ignore[union-attr]
    return {
        "schema_version": ChunkPlanRecord.SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "dataset_hash": sha256_tree(ds, DATASET_HASH_PATTERNS),
        "centerline_file": manifest.centerline.file,  # type: ignore[union-attr]
        "centerline_sha256": sha256_file(cl_file),
        "axis_range_m": [s0, s1],
        "policy": {
            "core_length_m": float(core_length_m),
            "overlap_m": float(overlap_m),
            "ownership_rule": OWNERSHIP_RULE,
        },
        "chunks": chunks,
        "unplaced_groups": unplaced,
    }


def build_chunk_plan(
    dataset_dir: str | Path,
    core_length_m: float,
    overlap_m: float,
    out_dir: str | Path | None = None,
) -> tuple[ChunkPlanRecord, Path]:
    """Plan, write once (or find the identical plan already written), verify, return."""
    ds = Path(dataset_dir)
    content = plan_content(ds, core_length_m, overlap_m)
    digest = _digest(_content_of(content))
    plan_id = f"cplan_{digest[:12]}"
    record = ChunkPlanRecord.from_dict(
        {
            **content,
            "plan_id": plan_id,
            "plan_digest": digest,
            "provenance": stamp(content["policy"], parents=[content["dataset_id"]]).model_dump(
                mode="json"
            ),
        }
    )
    out = Path(out_dir) if out_dir is not None else ds / PLAN_DIR / plan_id
    path = out / PLAN_FILE
    if path.is_file():
        existing = load_chunk_plan(path)
        if existing.plan_digest != digest:
            raise _refuse(f"{path} holds another plan ({existing.plan_id}); a plan is written once")
        return verify_chunk_plan(ds, path), path
    if out.exists() and any(out.iterdir()):
        raise _refuse(f"{out} is not empty and holds no {PLAN_FILE}")
    partial = out.parent / f".{out.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    record.save(partial / PLAN_FILE)
    if out.exists():
        out.rmdir()
    os.replace(partial, out)
    return verify_chunk_plan(ds, path), path


# ---------------------------------------------------------------- verification


def load_chunk_plan(path: str | Path) -> ChunkPlanRecord:
    p = Path(path)
    if p.is_dir():
        p = p / PLAN_FILE
    if not p.is_file():
        raise _refuse(f"{p} does not exist")
    try:
        return ChunkPlanRecord.load(p)
    except ValidationError as e:  # pragma: no cover - VersionedModel.load wraps these
        raise _refuse(f"{p}: {e}") from e


def check_invariants(rec: ChunkPlanRecord) -> None:
    """The structural rules a plan must satisfy whatever made it (contract §5.2)."""
    pol = rec.policy
    check_policy(pol.core_length_m, pol.overlap_m)
    if pol.ownership_rule != OWNERSHIP_RULE:
        raise _refuse(f"ownership rule {pol.ownership_rule!r} is not {OWNERSHIP_RULE!r}")
    if not rec.chunks:
        raise _refuse("the plan has no chunks")
    ids = [c.chunk_id for c in rec.chunks]
    if len(set(ids)) != len(ids):
        raise _refuse(f"chunk ids repeat: {ids}")
    if [c.ordinal for c in rec.chunks] != list(range(len(rec.chunks))):
        raise _refuse("chunk ordinals are not 0..n-1 in order")
    s0, s1 = rec.axis_range_m
    first, last = rec.chunks[0].core_range_m, rec.chunks[-1].core_range_m
    if abs(first[0] - s0) > EPS_M or abs(last[1] - s1) > EPS_M:
        raise _refuse(
            f"cores span [{first[0]}, {last[1]}] but the axis is [{s0}, {s1}]; a core outside "
            "the axis, or an axis end no core owns"
        )
    for a, b in zip(rec.chunks, rec.chunks[1:], strict=False):
        lo_a, hi_a = a.core_range_m
        lo_b, hi_b = b.core_range_m
        if not (lo_a < hi_a and lo_b < hi_b):
            raise _refuse(f"empty or reversed core in {a.chunk_id}/{b.chunk_id}")
        if hi_a < lo_b - EPS_M:
            raise _refuse(
                f"gap between {a.chunk_id} core [{lo_a}, {hi_a}] and {b.chunk_id} core "
                f"[{lo_b}, {hi_b}]: chainage {hi_a}-{lo_b} m would belong to no chunk"
            )
        if hi_a > lo_b + EPS_M:
            raise _refuse(
                f"{a.chunk_id} core [{lo_a}, {hi_a}] and {b.chunk_id} core [{lo_b}, {hi_b}] "
                "overlap: the same chainage would be owned twice"
            )
        if lo_b < lo_a:
            raise _refuse("cores are not in chainage order")
    for c in rec.chunks:
        lo, hi = c.core_range_m
        slo, shi = c.support_range_m
        if slo > lo + EPS_M or shi < hi - EPS_M:
            raise _refuse(f"{c.chunk_id} support [{slo}, {shi}] does not contain its core")
        want = (max(s0, lo - pol.overlap_m), min(s1, hi + pol.overlap_m))
        if abs(slo - want[0]) > EPS_M or abs(shi - want[1]) > EPS_M:
            raise _refuse(
                f"{c.chunk_id} support [{slo}, {shi}] is not its core widened by the overlap "
                f"{pol.overlap_m} m ({want})"
            )


def verify_chunk_plan(
    dataset_dir: str | Path, plan: str | Path | ChunkPlanRecord, *, manifest: Manifest | None = None
) -> ChunkPlanRecord:
    """Re-derive the plan from the dataset as it is now; refuse on the first difference."""
    ds = Path(dataset_dir)
    rec = plan if isinstance(plan, ChunkPlanRecord) else load_chunk_plan(plan)
    check_invariants(rec)
    if _digest(rec.content()) != rec.plan_digest:
        raise _refuse(f"{rec.plan_id}: the content does not hash to its plan_digest")
    if rec.plan_id != f"cplan_{rec.plan_digest[:12]}":
        raise _refuse(f"plan_id {rec.plan_id} is not derived from its digest")
    manifest = manifest or Manifest.load_dataset(ds)
    if rec.dataset_id != manifest.dataset_id:
        raise _refuse(
            f"{rec.plan_id} was made for dataset {rec.dataset_id}, not {manifest.dataset_id}"
        )
    want = plan_content(ds, rec.policy.core_length_m, rec.policy.overlap_m, manifest=manifest)
    if want["centerline_file"] != rec.centerline_file or (
        want["centerline_sha256"] != rec.centerline_sha256
    ):
        raise _refuse(
            f"{rec.plan_id} was cut along {rec.centerline_file} ({rec.centerline_sha256[:12]}), "
            f"the dataset's axis is now {want['centerline_file']} "
            f"({want['centerline_sha256'][:12]}); chainage means something else"
        )
    if want["dataset_hash"] != rec.dataset_hash:
        raise _refuse(
            f"{rec.plan_id} was made against dataset hash {rec.dataset_hash[:12]}, the dataset "
            f"now hashes to {want['dataset_hash'][:12]}: its groups, images or axis may have moved"
        )
    if canonical_json(want) != canonical_json(rec.content()):
        raise _refuse(f"{rec.plan_id} is not the plan this dataset and policy produce")
    return rec


__all__ = [
    "EPS_M",
    "OWNERSHIP_RULE",
    "PLAN_DIR",
    "PLAN_FILE",
    "ChunkPlanRecord",
    "ChunkPolicy",
    "PlannedChunk",
    "build_chunk_plan",
    "check_invariants",
    "check_policy",
    "core_ranges",
    "load_chunk_plan",
    "owner_index",
    "plan_content",
    "verify_chunk_plan",
]
