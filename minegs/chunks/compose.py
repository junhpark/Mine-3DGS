"""``ChunkRunSet`` — the runs of one plan put back together on the axis (Phase 5 §7).

Nothing here makes geometry. Each chunk was trained, rendered, surfaced and sectioned by the
existing path; this module checks that the pieces belong to one dataset, one plan and one
profile, and then reads each station of the axis from the one chunk that owns it
(``ChunkPlanRecord.owner_index``). Overlap is context for training and a diagnostic here, never
a second contribution: every station appears once, so nothing is integrated twice.

Every number is ``compare_to_reference``'s own: the stitched series against the scanned reference
over the declared holdout (and, diagnostically, over the whole core extent), each chunk's core
against the same reference, and adjacent chunks against each other over their shared support.
That last one is agreement, not accuracy — two chunks can agree and both be wrong.

Gaussian models are indexed, not merged: no averaging, deduplication or blending of overlaps, and
no transform is estimated between chunks — they were trained in one LOCAL_METRIC frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel, canonical_json
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import ProvenanceRecord, sha256_file, sha256_tree, stamp

CHUNK_RUN_SET_FILE = "chunk_run_set.json"
MATURITY = (
    "Structural composition of the chunk runs of one plan. No accuracy, seam quality, optimal "
    "chunk size or long-tunnel validation is claimed; G3: PENDING."
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkEntry(_Strict):
    chunk_id: str
    ordinal: int
    core_range_m: tuple[float, float]
    support_range_m: tuple[float, float]
    run_id: str | None = None
    status: str = "missing"
    surface_id: str | None = None
    sections_id: str | None = None
    #: The chunk's own Gaussian model, indexed (not merged).
    model: dict[str, Any] | None = None
    #: What the run recorded about its chunk: init/depth selection, scene_scale.
    run_chunk: dict[str, Any] | None = None
    real_gpu: bool = False
    owned_stations: int = 0


class ChunkRunSet(VersionedModel):
    """``chunk_run_set.json`` (docs/PHASE5_CONTRACT.md §7)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    chunk_set_id: str = Field(min_length=1)
    plan_id: str
    plan_digest: str
    dataset_id: str
    dataset_hash: str
    profile: dict[str, Any]
    frame: str = "LOCAL_METRIC"
    chunks: list[ChunkEntry]
    complete: bool
    requested_core_length_m: float
    completed_core_length_m: float
    coverage_fraction: float
    missing_chunks: list[str] = Field(default_factory=list)
    missing_intervals_m: list[tuple[float, float]] = Field(default_factory=list)
    stitched: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    seams: list[dict[str, Any]] = Field(default_factory=list)
    real_execution: bool = False
    g3_status: str = "PENDING"
    maturity_statement: str = MATURITY
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord


@dataclass
class ChunkInputs:
    """One chunk: its run, and the sections cut from that run's surface on the whole-axis grid."""

    run_dir: Path
    sections: Any  # SectionRecord; None only for a run that did not succeed


def _refuse(why: str) -> ContractError:
    return ContractError(f"chunk run set: {why}")


def _paired_summary(p) -> dict[str, Any]:
    s, v = p.sections, p.volume
    return {
        "requested_intervals_m": [list(r) for r in s.requested_intervals_m],
        "station_count": s.station_count,
        "paired_valid_count": s.paired_valid_count,
        "missing_prediction_count": s.missing_prediction_count,
        "missing_reference_count": s.missing_reference_count,
        "median_absolute_error_m2": s.median_absolute_error_m2,
        "p95_absolute_error_m2": s.p95_absolute_error_m2,
        "mean_signed_error_m2": s.mean_signed_error_m2,
        "volume": {
            "predicted_volume_m3": v.predicted_volume_m3,
            "reference_volume_m3": v.reference_volume_m3,
            "absolute_error_m3": v.absolute_error_m3,
            "relative_error": v.relative_error,
            "coverage_fraction": v.coverage_fraction,
            "integrated_intervals_m": [list(r) for r in v.integrated_intervals_m],
            "missing_intervals_m": [list(r) for r in v.missing_intervals_m],
        },
    }


def _empty_section(like) -> Any:
    """A station no present chunk owns: unobserved, never zero."""
    return like.model_copy(
        update={
            "area_m2": None,
            "valid": False,
            "n_points": 0,
            "empty_bins": len(like.radii_m),
            "radii_m": [None] * len(like.radii_m),
        }
    )


def _check_run(rec, label: str, dataset_id: str, dataset_hash: str) -> None:
    from minegs.eval.surface.render import require_metric_outputs

    if rec.dataset_id != dataset_id or rec.dataset_hash != dataset_hash:
        raise _refuse(
            f"{label} run {rec.run_id} was trained on {rec.dataset_id} ({rec.dataset_hash[:12]}), "
            f"not on this dataset as it is now ({dataset_id}, {dataset_hash[:12]})"
        )
    if rec.frame_of_outputs != "LOCAL_METRIC":
        raise _refuse(f"{label} run {rec.run_id} outputs are in {rec.frame_of_outputs}")
    require_metric_outputs(rec)


def stitch_sections(plan, sections: dict[str, Any], reference) -> tuple[Any, dict[str, int]]:
    """Every station of the reference grid, read from the one chunk that owns it.

    *sections* maps a chunk id to that chunk's sections, cut on the reference grid. A chunk that
    is not in it is missing: the stations it owns come back unobserved (area ``None``) — never
    zero, and never borrowed from a neighbour's overlap. Returns the stitched series (with the
    identity fields ``compare_to_reference`` reads) and how many stations each chunk owns.
    """
    from minegs.chunks.plan import EPS_M
    from minegs.eval.sections.sections import SectionSeries

    ref_series = reference.series
    ref_sorted = sorted(ref_series.sections, key=lambda x: x.chainage_m)
    stations = np.array([x.chainage_m for x in ref_sorted], dtype=np.float64)
    if len(stations) > 1 and np.any(np.diff(stations) <= EPS_M):
        raise _refuse("the reference grid repeats a chainage")
    lo, hi = plan.core_extent()
    outside = stations[(stations < lo - EPS_M) | (stations > hi + EPS_M)]
    if len(outside):
        raise _refuse(
            f"stations {outside[:3].tolist()} lie outside every chunk's core ({lo:g}-{hi:g} m); "
            "no chunk owns them"
        )
    unknown = sorted(set(sections) - {c.chunk_id for c in plan.chunks})
    if unknown:
        raise _refuse(f"chunks {unknown} are not in plan {plan.plan_id}")
    aligned: dict[str, list] = {}
    for cid, sec in sections.items():
        own = sorted(sec.series.sections, key=lambda x: x.chainage_m)
        got = np.array([x.chainage_m for x in own], dtype=np.float64)
        if len(got) != len(stations) or (
            len(got) and float(np.max(np.abs(got - stations))) > EPS_M
        ):
            raise _refuse(
                f"chunk {cid}'s sections ({len(got)} stations) are not the reference grid "
                f"({len(stations)} stations): a station is missing, moved or repeated"
            )
        aligned[cid] = own
    owner = plan.owner_index(stations)
    counts = {c.chunk_id: 0 for c in plan.chunks}
    out = []
    for k, ref_section in enumerate(ref_sorted):
        cid = plan.chunks[int(owner[k])].chunk_id
        counts[cid] += 1
        out.append(aligned[cid][k] if cid in aligned else _empty_section(ref_section))
    got = np.array([x.chainage_m for x in out], dtype=np.float64)
    if len(got) > 1 and not np.all(np.diff(got) > 0):
        raise _refuse("stitched chainages are not strictly increasing")
    series = SectionSeries(
        frame=ref_series.frame,
        interval_m=ref_series.interval_m,
        thickness_m=ref_series.thickness_m,
        angle_bins=ref_series.angle_bins,
        start_chainage_m=ref_series.start_chainage_m,
        end_chainage_m=ref_series.end_chainage_m,
        sections=out,
    )
    stitched = SimpleNamespace(
        reference_axis=reference.reference_axis,
        dataset_id=reference.dataset_id,
        dataset_hash=reference.dataset_hash,
        frame=reference.frame,
        series=series,
    )
    return stitched, counts


def compose_chunk_set(
    dataset_dir: str | Path,
    chunk_plan: str | Path,
    inputs: list[ChunkInputs],
    reference,
    *,
    chunk_set_id: str,
    ranges: list[tuple[float, float]] | None = None,
    allow_incomplete: bool = False,
) -> ChunkRunSet:
    from minegs.chunks.plan import verify_chunk_plan
    from minegs.eval.geometry.evaluate import load_dataset_and_centerline
    from minegs.eval.protocol import judge
    from minegs.eval.sections.models import check_section_record
    from minegs.eval.volume.coverage import merge_intervals
    from minegs.eval.volume.paired import compare_to_reference, require_same_grid
    from minegs.train.runner.base import (
        DATASET_HASH_PATTERNS,
        RunStatus,
        load_record,
        real_gpu_evidence,
    )

    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds, strict_layout=False)
    dataset_hash = sha256_tree(ds, DATASET_HASH_PATTERNS)
    plan = verify_chunk_plan(ds, chunk_plan, manifest=manifest)
    _, centerline = load_dataset_and_centerline(ds)  # TLS_GLOBAL, like the sections

    # ---- the reference: a scanned cloud of this dataset
    check_section_record(reference, manifest.dataset_id, dataset_hash, ds, manifest, centerline)
    if reference.source.kind != "raw_cloud":
        raise _refuse(
            f"the reference was cut from a {reference.source.kind}, not a scanned cloud; a "
            "reconstruction measured against a reconstruction has no error to report"
        )

    # ---- each chunk: its run, of this plan, with its own sections on the reference grid
    entries = {
        c.chunk_id: ChunkEntry(
            chunk_id=c.chunk_id,
            ordinal=c.ordinal,
            core_range_m=c.core_range_m,
            support_range_m=c.support_range_m,
        )
        for c in plan.chunks
    }
    present: dict[str, tuple[Any, Any]] = {}
    profile = None
    run_ids: set[str] = set()
    for item in inputs:
        rec = load_record(item.run_dir)
        label = f"chunk input {Path(item.run_dir).name}"
        _check_run(rec, label, manifest.dataset_id, dataset_hash)
        if rec.chunk is None or rec.chunk_id is None:
            raise _refuse(f"run {rec.run_id} is not a chunk of a plan")
        if rec.chunk.get("chunk_id") != rec.chunk_id:
            raise _refuse(
                f"run {rec.run_id} names chunk {rec.chunk_id} but its binding is for "
                f"{rec.chunk.get('chunk_id')}"
            )
        if rec.chunk.get("plan_digest") != plan.plan_digest:
            raise _refuse(
                f"run {rec.run_id} trained chunk {rec.chunk_id} of plan {rec.chunk.get('plan_id')}, "
                f"not of {plan.plan_id}"
            )
        planned = plan.chunk(rec.chunk_id)
        if (
            tuple(rec.chunk.get("core_range_m") or ()) != planned.core_range_m
            or tuple(rec.chunk.get("support_range_m") or ()) != planned.support_range_m
        ):
            raise _refuse(f"run {rec.run_id} records another core/support for {rec.chunk_id}")
        if rec.run_id in run_ids:
            raise _refuse(f"run {rec.run_id} is given twice")
        run_ids.add(rec.run_id)
        if rec.chunk_id in present or entries[rec.chunk_id].run_id is not None:
            raise _refuse(f"chunk {rec.chunk_id} is given twice")
        if profile is None:
            profile = rec.profile
        elif canonical_json(rec.profile) != canonical_json(profile):
            raise _refuse(
                f"run {rec.run_id} trained with profile {rec.profile.get('name')!r} settings "
                f"different from the others ({profile.get('name')!r}); one set is one experiment"
            )
        entry = entries[rec.chunk_id]
        entry.run_id = rec.run_id
        entry.status = rec.status.value
        entry.run_chunk = {
            k: v for k, v in rec.chunk.items() if k not in ("images", "capture_groups")
        }
        if rec.status is not RunStatus.SUCCEEDED:
            if not allow_incomplete:
                raise _refuse(
                    f"chunk {rec.chunk_id} run {rec.run_id} is {rec.status.value}; a chunk set "
                    "with a failed chunk is not complete (allow_incomplete reports it as missing)"
                )
            continue
        sec = item.sections
        if sec is None:
            raise _refuse(
                f"chunk {rec.chunk_id} run {rec.run_id} succeeded, but no sections cut from its "
                "surface were given"
            )
        check_section_record(sec, manifest.dataset_id, dataset_hash, ds, manifest, centerline)
        src = sec.source
        if src.kind != "surface" or src.run_id != rec.run_id:
            raise _refuse(
                f"the sections given for {rec.chunk_id} come from {src.kind} "
                f"{src.run_id or src.point_path!r}, not from the surface of run {rec.run_id}"
            )
        if src.point_sha256 == reference.source.point_sha256:
            raise _refuse(f"{rec.chunk_id}'s sections were cut from the reference cloud")
        require_same_grid(sec, reference)
        entry.surface_id = src.surface_id
        entry.sections_id = sec.section_id
        entry.real_gpu = real_gpu_evidence(rec)
        model = Path(item.run_dir) / rec.final_model if rec.final_model else None
        entry.model = {
            "path": rec.final_model,
            "sha256": sha256_file(model) if model is not None and model.is_file() else None,
            "gaussian_count": rec.gaussian_count,
        }
        present[rec.chunk_id] = (rec, sec)

    if profile is None or not present:
        raise _refuse("no chunk run succeeded; there is nothing to compose")
    missing = [c.chunk_id for c in plan.chunks if c.chunk_id not in present]
    if missing and not allow_incomplete:
        raise _refuse(
            f"chunks {missing} have no succeeded run; the set is not complete (allow_incomplete "
            "reports the missing core as missing, never as zero)"
        )

    # ---- stitch: every reference station from the one chunk that owns it
    stitched, counts = stitch_sections(
        plan, {cid: sec for cid, (_rec, sec) in present.items()}, reference
    )
    for cid, n in counts.items():
        entries[cid].owned_stations = n

    # ---- coverage of the core extent
    lo_all, hi_all = plan.core_extent()
    requested = hi_all - lo_all
    missing_iv = merge_intervals([entries[c].core_range_m for c in missing])
    completed = requested - sum(hi - lo for lo, hi in missing_iv)

    # ---- evaluation, every number from compare_to_reference
    notes: list[str] = []
    holdout = sorted(tuple(r) for r in judge(manifest).holdout_ranges_m)
    evaluation: dict[str, Any] = {"holdout": None, "extent": None, "per_chunk": {}}
    if ranges is not None:
        notes.append(
            "explicit evaluation ranges were given: the stitched result over them is "
            "diagnostic, not a held-out measurement"
        )
        evaluation["holdout"] = {
            "ranges_m": [list(r) for r in ranges],
            "diagnostic": True,
            **_paired_summary(compare_to_reference(stitched, reference, list(ranges))),
        }
    elif holdout:
        evaluation["holdout"] = {
            "ranges_m": [list(r) for r in holdout],
            "diagnostic": False,
            **_paired_summary(compare_to_reference(stitched, reference, holdout)),
        }
    else:
        notes.append("the dataset declares no geometry holdout; only diagnostic evaluations")
    evaluation["extent"] = {
        "ranges_m": [[lo_all, hi_all]],
        "diagnostic": True,
        **_paired_summary(compare_to_reference(stitched, reference, [(lo_all, hi_all)])),
    }
    for cid, (_rec, sec) in present.items():
        core = entries[cid].core_range_m
        evaluation["per_chunk"][cid] = {
            "ranges_m": [list(core)],
            "diagnostic": True,
            **_paired_summary(compare_to_reference(sec, reference, [core])),
        }
    notes.append(
        "extent and per-chunk evaluations include geometry the runs trained on (the init and "
        "depth come from the same survey); only the holdout evaluation is about held-out "
        "geometry. Per-chunk evaluations each include their closed core end, so their sums are "
        "not the stitched totals."
    )

    # ---- seams: adjacent chunks over their shared support (agreement, not accuracy)
    seams = []
    for a, b in zip(plan.chunks, plan.chunks[1:], strict=False):
        lo = max(a.support_range_m[0], b.support_range_m[0])
        hi = min(a.support_range_m[1], b.support_range_m[1])
        seam: dict[str, Any] = {
            "left": a.chunk_id,
            "right": b.chunk_id,
            "overlap_m": [lo, hi],
            "n_paired_sections": None,
            "median_abs_area_difference_m2": None,
            "p95_abs_area_difference_m2": None,
            "mean_signed_area_difference_m2": None,
        }
        if a.chunk_id in present and b.chunk_id in present and hi - lo > 0:
            p = compare_to_reference(present[b.chunk_id][1], present[a.chunk_id][1], [(lo, hi)])
            seam.update(
                n_paired_sections=p.sections.paired_valid_count,
                median_abs_area_difference_m2=p.sections.median_absolute_error_m2,
                p95_abs_area_difference_m2=p.sections.p95_absolute_error_m2,
                mean_signed_area_difference_m2=p.sections.mean_signed_error_m2,
            )
        seams.append(seam)
    notes.append(
        "seams compare two reconstructions with each other (right minus left over the shared "
        "support): agreement between chunks, not accuracy against the reference"
    )

    real = not missing and all(e.real_gpu for e in entries.values())
    if not real:
        notes.append(
            "not every chunk run recorded a GPU running the pinned upstream (or a chunk is "
            "missing): this set is structural evidence about the pipeline, not a measurement of "
            "a mine"
        )
    return ChunkRunSet(
        chunk_set_id=chunk_set_id,
        plan_id=plan.plan_id,
        plan_digest=plan.plan_digest,
        dataset_id=manifest.dataset_id,
        dataset_hash=dataset_hash,
        profile=profile,
        chunks=[entries[c.chunk_id] for c in plan.chunks],
        complete=not missing,
        requested_core_length_m=float(requested),
        completed_core_length_m=float(completed),
        coverage_fraction=float(completed / requested) if requested > 0 else 0.0,
        missing_chunks=missing,
        missing_intervals_m=[tuple(r) for r in missing_iv],
        stitched={
            "station_count": len(stitched.series.sections),
            "owners": [[cid, n] for cid, n in counts.items()],
            "grid": require_same_grid(stitched, reference),
            "reference_section_id": reference.section_id,
        },
        evaluation=json.loads(json.dumps(evaluation)),
        seams=seams,
        real_execution=real,
        notes=notes,
        provenance=stamp(
            {"plan_id": plan.plan_id, "chunk_set_id": chunk_set_id},
            parents=[manifest.dataset_id, plan.plan_id, *sorted(run_ids)],
        ),
    )


__all__ = [
    "CHUNK_RUN_SET_FILE",
    "MATURITY",
    "ChunkEntry",
    "ChunkInputs",
    "ChunkRunSet",
    "compose_chunk_set",
    "stitch_sections",
]
