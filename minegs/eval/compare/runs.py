"""Baseline against advanced, on one dataset (Phase 4 C3, docs/PHASE4_CONTRACT.md AD-12).

The comparison G3 will need is "two training configurations, same survey, same question". This
module makes it possible to ask and, before that, decides whether it can be asked at all. Two
runs are compared only if every one of these holds, and the first one that does not is a
refusal:

* **one dataset**: both runs were trained on this dataset as it is now (id and hash), and
  their sections were cut from it;
* **one protocol**: the evaluation ranges are the dataset's declared holdout, the same for both
  by construction, and any geometry or render reports were measured over the same thing;
* **one metric frame**: both runs' outputs are ``LOCAL_METRIC`` with an identity
  ``T_local_from_internal``;
* **one support**: both predictions are on the same section grid, against the same reference,
  and every number below is over the chainage *both* observed.

What it reports, it copies. Sections and volume come from ``compare_to_reference``, called
again on the common domain. Geometry, render, runtime and memory are read from the artifacts
that measured them. A metric one side lacks is ``null`` on that side and in the difference, not
0. Training loss is not read at all: a lower loss is not a better tunnel, and with depth
supervision the two losses are not even the same function.

There is no verdict field. Signed differences (advanced minus baseline) are stated, and
``G3: PENDING`` holds until real runs on a real survey are compared (§12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import ProvenanceRecord, sha256_tree, stamp
from minegs.eval.volume.paired import Interval, compare_to_reference, require_same_grid

RUN_COMPARISON_FILE = "run_comparison.json"
#: Every side must declare both, true, before the comparison can be called real.
REQUIRED_EXECUTION = ("real_gpu_execution", "real_renderer_execution")
MATURITY = (
    "Phase 4 baseline-vs-advanced comparison: structural. G3: PENDING. No improvement is "
    "claimed; differences are reported as measured, advanced minus baseline."
)
#: The numeric metrics a side reports, by group; differences are taken key by key.
METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "sections": (
        "median_absolute_error_m2",
        "mean_absolute_error_m2",
        "p95_absolute_error_m2",
        "mean_signed_error_m2",
        "paired_valid_count",
    ),
    "volume": (
        "predicted_volume_m3",
        "reference_volume_m3",
        "absolute_error_m3",
        "relative_error",
        "coverage_fraction",
    ),
    "geometry": (
        "accuracy_median_m",
        "accuracy_p95_m",
        "completeness_median_m",
        "completeness_p95_m",
        "chamfer_m",
    ),
    "render": ("psnr", "ssim", "lpips"),
    "runtime": ("train_seconds", "duration_s"),
    "memory": ("peak_gpu_memory_gb",),
    "model": ("gaussian_count",),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSide(_Strict):
    label: str
    run_id: str
    profile: str | None
    #: What the run asked for and what it was, from its own record.
    configuration: dict[str, Any] = Field(default_factory=dict)
    #: Per group, per key; ``None`` where the side has no such measurement.
    metrics: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    execution: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RunComparison(VersionedModel):
    """``run_comparison.json``."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    comparison_id: str = Field(min_length=1)
    dataset_id: str
    dataset_hash: str
    frame: str = "LOCAL_METRIC"
    protocol: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, Any] = Field(default_factory=dict)
    requested_intervals_m: list[Interval] = Field(default_factory=list)
    common_intervals_m: list[Interval] = Field(default_factory=list)
    common_length_m: float = 0.0
    excluded_intervals_m: list[Interval] = Field(default_factory=list)
    baseline: RunSide
    advanced: RunSide
    #: advanced minus baseline, per group and key; ``None`` unless both sides measured it.
    differences: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    real_execution: bool = False
    g3_status: str = "PENDING"
    maturity_statement: str = MATURITY
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord


@dataclass
class RunInputs:
    """One side: its run, its sections and reference, and whatever else measured it."""

    run_dir: Path
    sections_predicted: Any  # SectionRecord
    sections_reference: Any  # SectionRecord
    geometry: dict[str, Any] | None = None  # GeometryReport, as JSON
    render: dict[str, Any] | None = None  # RenderReport, as JSON
    e2e_report: dict[str, Any] | None = None  # Phase2Report, as JSON


# ---------------------------------------------------------------- helpers


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def _subtract(whole: list[Interval], part: list[Interval]) -> list[Interval]:
    out = []
    for lo, hi in whole:
        cuts = sorted((max(lo, a), min(hi, b)) for a, b in part if min(hi, b) > max(lo, a))
        cursor = lo
        for a, b in cuts:
            if a > cursor:
                out.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < hi:
            out.append((cursor, hi))
    return out


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool) or v is None:
        return None
    return v if isinstance(v, (int, float)) else None


def _check_run(rec, label: str, manifest: Manifest, dataset_hash: str) -> None:
    from minegs.eval.surface.render import require_metric_outputs
    from minegs.train.runner.base import RunStatus

    if rec.status is not RunStatus.SUCCEEDED:
        raise ContractError(f"the {label} run {rec.run_id} is {rec.status.value}, not succeeded")
    if rec.dataset_id != manifest.dataset_id or rec.dataset_hash != dataset_hash:
        raise ContractError(
            f"the {label} run {rec.run_id} was trained on dataset {rec.dataset_id} "
            f"({rec.dataset_hash[:12]}), not {manifest.dataset_id} as it is now "
            f"({dataset_hash[:12]}); two runs on two datasets are two experiments"
        )
    if rec.frame_of_outputs != "LOCAL_METRIC":
        raise ContractError(f"the {label} run's outputs are in {rec.frame_of_outputs}")
    require_metric_outputs(rec)


def _configuration(rec) -> dict[str, Any]:
    prof = rec.profile or {}
    cfg = rec.trainer_config or {}
    sup = rec.depth_supervision or None
    return {
        "profile": prof.get("name"),
        "requests": dict(prof.get("requests") or {}),
        "data_factor": prof.get("data_factor"),
        "max_steps": prof.get("max_steps"),
        "strategy": cfg.get("strategy") or (prof.get("backend_args") or {}).get("strategy"),
        "trainer_entrypoint": (rec.trainer or {}).get("entrypoint"),
        "app_opt": cfg.get("app_opt"),
        "depth_supervision": None
        if sup is None
        else {
            k: sup.get(k)
            for k in ("supervision_id", "artifact_sha256", "source_kind", "confidence_semantics")
        },
        "optimised_images": rec.optimised_images,
    }


def _execution(report: dict[str, Any] | None) -> dict[str, bool]:
    """Real-execution flags, read from the side's own Phase 2 report and nowhere else."""
    if report is None:
        return {}
    return {
        "real_gpu_execution": bool((report.get("training") or {}).get("real_gpu_execution")),
        "real_renderer_execution": bool(
            (report.get("reconstruction") or {}).get("real_renderer_execution")
        ),
    }


def _same_measurement(a: dict | None, b: dict | None, keys: tuple[str, ...], what: str) -> None:
    if a is None or b is None:
        return
    for k in keys:
        if a.get(k) != b.get(k):
            raise ContractError(
                f"the two {what} reports were not measured the same way ({k}: {a.get(k)!r} vs "
                f"{b.get(k)!r}); their numbers would not be about the same question"
            )


# ---------------------------------------------------------------- the comparison


def compare_runs(
    dataset_dir: str | Path,
    baseline: RunInputs,
    advanced: RunInputs,
    *,
    comparison_id: str,
    ranges: list[Interval] | None = None,
) -> RunComparison:
    from minegs.eval.geometry.evaluate import load_dataset_and_centerline
    from minegs.eval.protocol import judge
    from minegs.eval.sections.models import check_section_record
    from minegs.train.runner.base import DATASET_HASH_PATTERNS, load_record

    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds, strict_layout=False)
    dataset_hash = sha256_tree(ds, DATASET_HASH_PATTERNS)
    recs = {}
    for label, side in (("baseline", baseline), ("advanced", advanced)):
        rec = load_record(side.run_dir)
        _check_run(rec, label, manifest, dataset_hash)
        recs[label] = rec
    if recs["baseline"].run_id == recs["advanced"].run_id:
        raise ContractError("the baseline and the advanced run are the same run")

    # ---- one protocol: the declared holdout, unless the caller names ranges explicitly
    j = judge(manifest)
    holdout = sorted(tuple(r) for r in j.holdout_ranges_m)
    if ranges is None:
        if not holdout:
            raise ContractError(
                f"{manifest.dataset_id} declares no geometry holdout; pass explicit ranges only "
                "if the comparison is meant to be diagnostic"
            )
        ranges = holdout
    ranges = sorted(tuple(r) for r in ranges)
    explicit_note = None
    if ranges != holdout:
        explicit_note = (
            "the comparison domain is not the dataset's declared holdout, so these numbers are "
            "diagnostic: they are not over held-out geometry"
        )

    # ---- one support: sections of these runs, of this dataset, on one grid, one reference
    _, centerline = load_dataset_and_centerline(ds)  # TLS_GLOBAL, like the sections
    grids = []
    for label, side in (("baseline", baseline), ("advanced", advanced)):
        pred, ref = side.sections_predicted, side.sections_reference
        for r in (pred, ref):
            check_section_record(r, manifest.dataset_id, dataset_hash, ds, manifest, centerline)
        src = pred.source
        if src.kind != "surface" or src.run_id != recs[label].run_id:
            raise ContractError(
                f"the {label} predicted sections come from {src.kind} "
                f"{src.run_id or src.point_path!r}, not from the surface of run "
                f"{recs[label].run_id}; they are not this run's sections"
            )
        grids.append(require_same_grid(pred, ref))
    if grids[0] != grids[1]:
        diff = sorted(k for k in grids[0] if grids[0][k] != grids[1].get(k))
        raise ContractError(f"the two runs' sections were cut on different grids ({diff})")
    ref_a = baseline.sections_reference.source.point_sha256
    ref_b = advanced.sections_reference.source.point_sha256
    if ref_a != ref_b:
        raise ContractError(
            "the two runs are measured against different references "
            f"({str(ref_a)[:12]} vs {str(ref_b)[:12]}); the errors would not share a truth"
        )
    _same_measurement(
        baseline.geometry,
        advanced.geometry,
        ("max_dist_m", "chainage_range_m", "frame", "claim"),
        "geometry",
    )
    _same_measurement(
        baseline.render, advanced.render, ("test_groups", "n_images", "claim"), "render"
    )

    # ---- the common domain, and both sides re-measured on it
    first = {
        label: compare_to_reference(side.sections_predicted, side.sections_reference, ranges)
        for label, side in (("baseline", baseline), ("advanced", advanced))
    }
    common = _intersect(
        first["baseline"].volume.integrated_intervals_m,
        first["advanced"].volume.integrated_intervals_m,
    )
    excluded = _subtract(list(ranges), common)
    paired = {
        label: compare_to_reference(side.sections_predicted, side.sections_reference, common)
        for label, side in (("baseline", baseline), ("advanced", advanced))
    }

    sides = {}
    for label, side in (("baseline", baseline), ("advanced", advanced)):
        rec = recs[label]
        p = paired[label]
        geo = side.geometry or {}
        metrics = {
            "sections": {k: _num(getattr(p.sections, k)) for k in METRIC_KEYS["sections"]},
            "volume": {k: _num(getattr(p.volume, k)) for k in METRIC_KEYS["volume"]},
            "geometry": {
                "accuracy_median_m": _num((geo.get("accuracy") or {}).get("median")),
                "accuracy_p95_m": _num((geo.get("accuracy") or {}).get("p95")),
                "completeness_median_m": _num((geo.get("completeness") or {}).get("median")),
                "completeness_p95_m": _num((geo.get("completeness") or {}).get("p95")),
                "chamfer_m": _num(geo.get("chamfer_m")),
            },
            "render": {k: _num((side.render or {}).get(k)) for k in METRIC_KEYS["render"]},
            "runtime": {
                "train_seconds": _num(rec.train_seconds),
                "duration_s": _num(rec.duration_s),
            },
            "memory": {"peak_gpu_memory_gb": _num(rec.peak_gpu_memory_gb)},
            "model": {"gaussian_count": _num(rec.gaussian_count)},
        }
        # coverage over what was requested, before restricting to the common domain
        metrics["volume"]["coverage_fraction"] = _num(first[label].volume.coverage_fraction)
        sides[label] = RunSide(
            label=label,
            run_id=rec.run_id,
            profile=(rec.profile or {}).get("name"),
            configuration=_configuration(rec),
            metrics=metrics,
            execution=_execution(side.e2e_report),
        )

    differences: dict[str, dict[str, float | None]] = {}
    for group, keys in METRIC_KEYS.items():
        differences[group] = {}
        for k in keys:
            a = sides["baseline"].metrics.get(group, {}).get(k)
            b = sides["advanced"].metrics.get(group, {}).get(k)
            differences[group][k] = None if a is None or b is None else float(b - a)

    notes: list[str] = [] if explicit_note is None else [explicit_note]
    real = True
    for label in ("baseline", "advanced"):
        ex = sides[label].execution
        missing = [k for k in REQUIRED_EXECUTION if k not in ex]
        substituted = sorted(k for k in REQUIRED_EXECUTION if ex.get(k) is False)
        if missing:
            notes.append(f"the {label} side did not report whether {', '.join(missing)} ran")
        if substituted:
            notes.append(f"the {label} side substituted {', '.join(substituted)}")
        real = real and not missing and not substituted
    if not real:
        notes.append(
            "so this comparison is structural evidence about the pipeline, not a measurement "
            "of a mine"
        )
    if not common:
        notes.append("the two runs share no observed chainage; no section/volume difference")
    if excluded:
        notes.append(f"{len(excluded)} requested interval(s) are outside the common domain")
    cb, ca = sides["baseline"].configuration, sides["advanced"].configuration
    if cb.get("data_factor") != ca.get("data_factor"):
        notes.append(
            f"the runs trained at data_factor {cb.get('data_factor')} and "
            f"{ca.get('data_factor')} and both were rendered at full resolution; the renderer's "
            "resolution gap (docs/PHASE4_CONTRACT.md §2.3-13) differs between them"
        )
    if cb.get("strategy") != ca.get("strategy"):
        notes.append(
            "the strategies differ; the mcmc preset also sets init_opa, init_scale, opacity_reg "
            "and scale_reg, so a difference is attributable to the whole preset"
        )
    if (ca.get("app_opt") or cb.get("app_opt")) and any(
        sides[x].metrics["render"]["psnr"] is not None for x in sides
    ):
        notes.append(
            "a side used appearance embedding: held-out views carry no trained embedding and "
            "are rendered with the zero embedding (upstream eval), so render metrics compare "
            "that policy too"
        )

    return RunComparison(
        comparison_id=comparison_id,
        dataset_id=manifest.dataset_id,
        dataset_hash=dataset_hash,
        protocol={
            "claims": [c.value for c in j.claims],
            "protocols": [p.value for p in j.protocols],
            "holdout_ranges_m": holdout,
        },
        grid=grids[0],
        requested_intervals_m=list(ranges),
        common_intervals_m=common,
        common_length_m=float(sum(hi - lo for lo, hi in common)),
        excluded_intervals_m=excluded,
        baseline=sides["baseline"],
        advanced=sides["advanced"],
        differences=differences,
        real_execution=real,
        notes=notes,
        provenance=stamp(
            {"comparison_id": comparison_id, "ranges": [list(r) for r in ranges]},
            parents=[recs["baseline"].run_id, recs["advanced"].run_id],
        ),
    )


__all__ = [
    "MATURITY",
    "METRIC_KEYS",
    "REQUIRED_EXECUTION",
    "RUN_COMPARISON_FILE",
    "RunComparison",
    "RunInputs",
    "RunSide",
    "compare_runs",
]
