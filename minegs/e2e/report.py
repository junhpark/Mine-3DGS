"""The Phase 2 report: an aggregation with no science of its own (§Phase 2 C5).

Every number here is copied from an artifact that already verified itself. The report does not
recompute a volume error, re-derive a claim or re-read a point cloud — it reads the ledger and
the JSON the stages left behind, and arranges it. That is deliberate: a report that recomputes
is a second implementation, and a second implementation is a second answer waiting to disagree
with the first in front of a reviewer.

``phase2_report.json`` is the source of truth and ``phase2_report.md`` is a rendering of *it*.
The Markdown does no arithmetic. Two documents that each work out the volume error are two
documents that can differ by a rounding decision nobody will be able to explain later.

Three things the report is not allowed to do, all of which would be easy and all of which would
be the whole problem:

* **Fill a gap.** A value it cannot find is ``null`` with a note saying why. An invented number
  travels further from a report than from anywhere else in this repository.
* **Upgrade a status.** ``real_data_validation_status`` stays ``not_validated`` and
  ``human_visual_review_status`` stays ``pending``, whatever the numbers look like. Nothing in
  minegs writes anything else there; a person does, after looking (§Phase 2 §16).
* **Confuse "ran" with "was executed for real".** A substituted trainer and a substituted
  renderer are recorded as substitutions, on the stage and in the report, because a structural
  gate that reads like a G2 is worse than no gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minegs.core.errors import ContractError
from minegs.core.provenance import make_id, stamp
from minegs.e2e.models import (
    REPORT_JSON_FILE,
    REPORT_MD_FILE,
    STAGE_ORDER,
    DatasetSummary,
    GeometrySummary,
    MaturitySummary,
    Phase2Report,
    ReconstructionSummary,
    RuntimeSummary,
    SectionSummary,
    SourceSummary,
    Stage,
    TrainingSummary,
    VolumeSummary,
    WorkflowState,
)
from minegs.e2e.runner import now_iso

__all__ = [
    "MATURITY_PENDING",
    "build_report",
    "render_markdown",
    "report_from_workflow",
    "write_report",
]

MATURITY_PENDING = (
    "Phase 2 E57 end-to-end workflow is implemented and structurally tested. "
    "Real-data scientific validation remains NOT VALIDATED. Phase 2 G2 remains PENDING."
)


def _out(state: WorkflowState, stage: Stage) -> dict[str, Any]:
    """A stage's outputs, or an empty mapping. Absence is a fact the report states."""
    rec = state.stages[stage]
    return dict(rec.outputs) if rec.usable else {}


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else {}


def build_report(state: WorkflowState) -> Phase2Report:
    """Aggregate the ledger and the artifacts it points at. Reads; never recomputes."""
    ingest, dataset = _out(state, Stage.INGEST), _out(state, Stage.DATASET)
    train, depth = _out(state, Stage.TRAIN), _out(state, Stage.DEPTH)
    surface, geometry = _out(state, Stage.SURFACE), _out(state, Stage.GEOMETRY)
    sv = _out(state, Stage.SECTIONS_VOLUME)
    paired = _load(sv.get("paired_validation_path"))

    return Phase2Report(
        report_id=make_id("phase2"),
        workflow_id=state.workflow_id,
        generated_at=now_iso(),
        source=_source(ingest),
        dataset=_dataset(dataset),
        training=_training(state, train),
        reconstruction=_reconstruction(depth, surface),
        geometry=_geometry(geometry),
        sections=_sections(paired, sv),
        volume=_volume(paired, sv),
        runtime=_runtime(state),
        maturity=_maturity(state, train, depth),
        stages=[_stage_row(state, s) for s in STAGE_ORDER],
        provenance=stamp(dict(state.config), parents=[state.workflow_id]),
    )


def _source(ingest: dict[str, Any]) -> SourceSummary:
    notes: list[str] = []
    if not ingest:
        notes.append("the ingest stage has not completed, so the survey is unidentified")
    elif ingest.get("adopted"):
        notes.append(
            "the staging tree was extracted elsewhere and adopted by this workflow; the E57 "
            "identity below is the one that extraction recorded, not one this run computed"
        )
    if ingest and not ingest.get("source_sha256"):
        notes.append("the source digest was not computed, so the survey has no identity here")
    return SourceSummary(
        file_name=ingest.get("source_file_name"),
        sha256=ingest.get("source_sha256"),
        size_bytes=ingest.get("source_size_bytes"),
        capture_metadata={
            k: ingest[k]
            for k in (
                "e57_writer",
                "coordinate_metadata",
                "scan_count_declared",
                "registration",
                "output_frame",
                "n_scans_extracted",
                "n_images_extracted",
                "n_images_skipped",
                "mapping_status_counts",
            )
            if ingest.get(k) is not None
        },
        notes=notes,
    )


def _dataset(dataset: dict[str, Any]) -> DatasetSummary:
    ranges = [tuple(r) for r in dataset.get("geometry_holdout_ranges_m", [])]
    notes = list(dataset.get("refusals") or [])
    if dataset and dataset.get("golden_gate_passed") is not True:
        notes.append(f"golden gate: {dataset.get('golden_gate_result')}")
    return DatasetSummary(
        dataset_id=dataset.get("dataset_id"),
        dataset_hash=dataset.get("dataset_hash"),
        station_count=dataset.get("n_stations"),
        image_count=dataset.get("n_images"),
        init_point_count=dataset.get("init_points"),
        evaluated_length_m=sum(hi - lo for lo, hi in ranges) if ranges else None,
        centerline_length_m=dataset.get("centerline_length_m"),
        geometry_holdout_ranges_m=ranges,
        camera_convention=dict(dataset.get("camera_convention") or {}),
        protocol=list(dataset.get("protocols") or []),
        claims_allowed=list(dataset.get("claims") or []),
        golden_gate_passed=dataset.get("golden_gate_passed"),
        notes=notes,
    )


def _training(state: WorkflowState, train: dict[str, Any]) -> TrainingSummary:
    rec = state.stages[Stage.TRAIN]
    env = dict(rec.runtime_env or {})
    notes: list[str] = []
    if env.get("reason"):
        notes.append(env["reason"])
    if train and not train.get("real_gpu_execution"):
        notes.append(
            "the training backend was substituted; these weights were not produced by a real "
            "gsplat run on a real GPU"
        )
    backend = dict(train.get("backend") or {})
    return TrainingSummary(
        run_id=train.get("run_id"),
        backend=backend.get("name"),
        backend_version=backend.get("version"),
        profile=train.get("profile_name"),
        steps=train.get("checkpoint_step"),
        checkpoint_step=train.get("checkpoint_step"),
        git_commit=rec.git_commit or None,
        gpu_model=env.get("gpu_model"),
        cuda_version=env.get("cuda_version"),
        runner=train.get("runner"),
        real_gpu_execution=bool(train.get("real_gpu_execution")),
        notes=notes,
    )


def _reconstruction(depth: dict[str, Any], surface: dict[str, Any]) -> ReconstructionSummary:
    notes: list[str] = []
    if depth and not depth.get("real_renderer_execution"):
        notes.append(
            "the depth renderer was substituted; GsplatDepthRenderer.render did not execute"
        )
    return ReconstructionSummary(
        depth_manifest_id=depth.get("depth_manifest_id"),
        renderer=depth.get("renderer"),
        renderer_version=depth.get("renderer_version"),
        depth_map_count=depth.get("depth_map_count"),
        mean_depth_valid_ratio=depth.get("mean_valid_ratio"),
        surface_id=surface.get("surface_id"),
        surface_point_count=surface.get("point_count"),
        depth_source=surface.get("depth_source"),
        real_renderer_execution=bool(depth.get("real_renderer_execution")),
        notes=notes,
    )


def _geometry(geometry: dict[str, Any]) -> GeometrySummary:
    rng = geometry.get("chainage_range_m")
    return GeometrySummary(
        claim=geometry.get("claim"),
        chainage_range_m=tuple(rng) if rng else None,
        max_dist_m=geometry.get("max_dist_m"),
        accuracy_median_m=geometry.get("accuracy_median_m"),
        accuracy_p95_m=geometry.get("accuracy_p95_m"),
        completeness_median_m=geometry.get("completeness_median_m"),
        completeness_p95_m=geometry.get("completeness_p95_m"),
        chamfer_m=geometry.get("chamfer_m"),
        f_score=dict(geometry.get("f_score") or {}),
        notes=list(geometry.get("notes") or []),
    )


def _sections(paired: dict[str, Any], sv: dict[str, Any]) -> SectionSummary:
    s = dict(paired.get("sections") or {})
    grid = dict(paired.get("grid") or sv.get("grid") or {})
    return SectionSummary(
        interval_m=grid.get("interval_m"),
        thickness_m=grid.get("thickness_m"),
        angle_bins=grid.get("angle_bins"),
        requested_ranges_m=[tuple(r) for r in s.get("requested_intervals_m", [])],
        station_count=s.get("station_count"),
        paired_valid_count=s.get("paired_valid_count"),
        missing_prediction_count=s.get("missing_prediction_count"),
        missing_reference_count=s.get("missing_reference_count"),
        mean_absolute_error_m2=s.get("mean_absolute_error_m2"),
        median_absolute_error_m2=s.get("median_absolute_error_m2"),
        p95_absolute_error_m2=s.get("p95_absolute_error_m2"),
        mean_signed_error_m2=s.get("mean_signed_error_m2"),
        mean_relative_error=s.get("mean_relative_error"),
        notes=list(s.get("notes") or []),
    )


def _volume(paired: dict[str, Any], sv: dict[str, Any]) -> VolumeSummary:
    v = dict(paired.get("volume") or {})
    notes = list(v.get("notes") or [])
    if v.get("relative_error") is None and v.get("relative_error_reason"):
        notes.append(v["relative_error_reason"])
    if sv.get("volume_accuracy_refusal"):
        notes.append(
            "a volume_accuracy claim is not available from these sections: "
            f"{sv['volume_accuracy_refusal']}"
        )
    return VolumeSummary(
        predicted_volume_m3=v.get("predicted_volume_m3"),
        reference_volume_m3=v.get("reference_volume_m3"),
        signed_error_m3=v.get("signed_error_m3"),
        absolute_error_m3=v.get("absolute_error_m3"),
        relative_error=v.get("relative_error"),
        requested_length_m=v.get("requested_length_m"),
        common_covered_length_m=v.get("common_covered_length_m"),
        coverage_fraction=v.get("coverage_fraction"),
        integrated_intervals_m=[tuple(i) for i in v.get("integrated_intervals_m", [])],
        missing_intervals_m=[tuple(i) for i in v.get("missing_intervals_m", [])],
        notes=notes,
    )


def _runtime(state: WorkflowState) -> RuntimeSummary:
    def secs(stage: Stage) -> float | None:
        return state.stages[stage].elapsed_seconds

    evaluation = [secs(Stage.GEOMETRY), secs(Stage.SECTIONS_VOLUME)]
    measured = [state.stages[s].elapsed_seconds for s in STAGE_ORDER]
    return RuntimeSummary(
        ingest_seconds=secs(Stage.INGEST),
        dataset_seconds=secs(Stage.DATASET),
        training_seconds=secs(Stage.TRAIN),
        depth_render_seconds=secs(Stage.DEPTH),
        surface_build_seconds=secs(Stage.SURFACE),
        evaluation_seconds=sum(x for x in evaluation if x is not None)
        if any(x is not None for x in evaluation)
        else None,
        # Summed over the stages that ran. A reused stage contributes nothing, which is the
        # honest reading: this workflow did not spend that time.
        total_seconds=sum(x for x in measured if x is not None)
        if any(x is not None for x in measured)
        else None,
    )


def _maturity(
    state: WorkflowState, train: dict[str, Any], depth: dict[str, Any]
) -> MaturitySummary:
    complete = all(state.stages[s].usable for s in STAGE_ORDER if s is not Stage.REPORT)
    notes: list[str] = []
    if not complete:
        notes.append(
            "not every stage completed: "
            + ", ".join(
                f"{s.value}={state.stages[s].status.value}"
                for s in STAGE_ORDER
                if not state.stages[s].usable and s is not Stage.REPORT
            )
        )
    if not train.get("real_gpu_execution"):
        notes.append("training was substituted, so no real GPU executed this workflow")
    if not depth.get("real_renderer_execution"):
        notes.append("depth was substituted, so GsplatDepthRenderer.render did not execute")
    return MaturitySummary(
        structural_status="implemented_and_structurally_tested" if complete else "incomplete",
        # Never set from here. A workflow cannot validate itself, and the value a human sets
        # after looking is the only thing that moves either of the two below.
        real_data_validation_status="not_validated",
        human_visual_review_status="pending",
        statement=MATURITY_PENDING,
        notes=notes,
    )


def _stage_row(state: WorkflowState, stage: Stage) -> dict[str, Any]:
    rec = state.stages[stage]
    return {
        "stage": stage.value,
        "status": rec.status.value,
        "elapsed_seconds": rec.elapsed_seconds,
        "input_fingerprint": rec.input_fingerprint,
        "git_commit": rec.git_commit,
        "minegs_version": rec.minegs_version,
        "failure_reason": rec.failure_reason,
    }


# ---------------------------------------------------------------- rendering


def _m(value: Any, unit: str = "", digits: int = 3) -> str:
    """One place where a null becomes a dash, so no table cell has to guess."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _mm(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.1f} mm"


def _intervals(items: list[tuple[float, float]]) -> str:
    return ", ".join(f"{lo:g}–{hi:g} m" for lo, hi in items) or "none"


def render_markdown(report: Phase2Report) -> str:
    """The same report, for a person. No arithmetic: every value is read from the model."""
    r = report
    lines: list[str] = [
        f"# Phase 2 E57 end-to-end report — {r.dataset.dataset_id or 'unidentified dataset'}",
        "",
        f"report `{r.report_id}` · workflow `{r.workflow_id}` · generated {r.generated_at}",
        "",
        "## Maturity",
        "",
        f"> {r.maturity.statement}",
        "",
        f"- structural status: **{r.maturity.structural_status}**",
        f"- real-data validation: **{r.maturity.real_data_validation_status}**",
        f"- human visual review: **{r.maturity.human_visual_review_status}**",
        f"- real GPU training executed: **{r.training.real_gpu_execution}**",
        f"- real depth renderer executed: **{r.reconstruction.real_renderer_execution}**",
    ]
    lines += [f"- note: {n}" for n in r.maturity.notes]
    lines += [
        "",
        "## Source",
        "",
        f"- file: `{r.source.file_name or '—'}`",
        f"- sha256: `{r.source.sha256 or '—'}`",
        f"- size: {_m(r.source.size_bytes)} bytes",
    ]
    lines += [f"- {k}: {v}" for k, v in sorted(r.source.capture_metadata.items())]
    lines += [f"- note: {n}" for n in r.source.notes]

    d = r.dataset
    lines += [
        "",
        "## Dataset",
        "",
        f"- dataset_id: `{d.dataset_id or '—'}`",
        f"- dataset_hash: `{(d.dataset_hash or '—')[:16]}`",
        f"- stations / images / init points: {_m(d.station_count)} / {_m(d.image_count)} / "
        f"{_m(d.init_point_count)}",
        f"- centerline length: {_m(d.centerline_length_m, ' m', 2)}",
        f"- evaluated (holdout) length: {_m(d.evaluated_length_m, ' m', 2)}",
        f"- geometry holdout: {_intervals(d.geometry_holdout_ranges_m)}",
        f"- camera convention: {d.camera_convention.get('label', '—')} "
        f"({d.camera_convention.get('source', '—')})",
        f"- protocol: {', '.join(d.protocol) or '—'}",
        f"- claims the dataset allows: {', '.join(d.claims_allowed) or '—'}",
        f"- golden gate: {_m(d.golden_gate_passed)}",
    ]
    lines += [f"- note: {n}" for n in d.notes]

    t = r.training
    lines += [
        "",
        "## Training",
        "",
        f"- run: `{t.run_id or '—'}` ({t.runner or '—'}, profile {t.profile or '—'})",
        f"- backend: {t.backend or '—'} {t.backend_version or ''}".rstrip(),
        f"- steps: {_m(t.steps)}",
        f"- GPU / CUDA: {t.gpu_model or '—'} / {t.cuda_version or '—'}",
        f"- minegs commit: `{t.git_commit or '—'}`",
    ]
    lines += [f"- note: {n}" for n in t.notes]

    c = r.reconstruction
    lines += [
        "",
        "## Reconstruction",
        "",
        f"- depth manifest: `{c.depth_manifest_id or '—'}` "
        f"({c.renderer or '—'} {c.renderer_version or ''})".rstrip(),
        f"- depth maps: {_m(c.depth_map_count)} (mean valid ratio "
        f"{_m(c.mean_depth_valid_ratio, '', 3)})",
        f"- surface: `{c.surface_id or '—'}` — {_m(c.surface_point_count)} points, "
        f"depth_source `{c.depth_source or '—'}`",
    ]
    lines += [f"- note: {n}" for n in c.notes]

    g = r.geometry
    lines += [
        "",
        "## Geometry (held-out TLS, both directions)",
        "",
        f"claim: **{g.claim or '—'}** over "
        f"{_intervals([g.chainage_range_m] if g.chainage_range_m else [])}, "
        f"max distance {_m(g.max_dist_m, ' m', 2)}",
        "",
        "| direction | median | P95 |",
        "|---|---|---|",
        f"| accuracy (pred → TLS) | {_mm(g.accuracy_median_m)} | {_mm(g.accuracy_p95_m)} |",
        f"| completeness (TLS → pred) | {_mm(g.completeness_median_m)} | "
        f"{_mm(g.completeness_p95_m)} |",
        "",
        f"Chamfer: {_mm(g.chamfer_m)}",
        "",
        "| F-score τ (m) | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.4f} |" for k, v in sorted(g.f_score.items(), key=lambda kv: kv[0])]
    lines += [f"- note: {n}" for n in g.notes]

    s = r.sections
    lines += [
        "",
        "## Sections (prediction vs held-out TLS, same grid)",
        "",
        f"grid: {_m(s.interval_m, ' m', 2)} stations, {_m(s.thickness_m, ' m', 2)} slab, "
        f"{_m(s.angle_bins)} bins over {_intervals(s.requested_ranges_m)}",
        "",
        f"- stations in range: {_m(s.station_count)}",
        f"- paired (both observed): {_m(s.paired_valid_count)}",
        f"- missing prediction / reference: {_m(s.missing_prediction_count)} / "
        f"{_m(s.missing_reference_count)}",
        f"- area error MAE / median / P95: {_m(s.mean_absolute_error_m2, ' m²')} / "
        f"{_m(s.median_absolute_error_m2, ' m²')} / {_m(s.p95_absolute_error_m2, ' m²')}",
        f"- mean signed error: {_m(s.mean_signed_error_m2, ' m²')}",
        f"- mean relative error: {_m(s.mean_relative_error, '', 4)}",
    ]
    lines += [f"- note: {n}" for n in s.notes]

    v = r.volume
    lines += [
        "",
        "## Volume (common integration domain only)",
        "",
        f"- predicted: {_m(v.predicted_volume_m3, ' m³', 2)}",
        f"- held-out TLS reference: {_m(v.reference_volume_m3, ' m³', 2)}",
        f"- signed / absolute error: {_m(v.signed_error_m3, ' m³', 2)} / "
        f"{_m(v.absolute_error_m3, ' m³', 2)}",
        f"- relative error: {_m(v.relative_error, '', 4)}",
        f"- coverage: {_m(v.common_covered_length_m, ' m', 2)} of "
        f"{_m(v.requested_length_m, ' m', 2)} ({_m(v.coverage_fraction, '', 3)})",
        f"- integrated: {_intervals(v.integrated_intervals_m)}",
        f"- missing: {_intervals(v.missing_intervals_m)}",
    ]
    lines += [f"- note: {n}" for n in v.notes]

    rt = r.runtime
    lines += [
        "",
        "## Runtime",
        "",
        "| stage | seconds |",
        "|---|---|",
        f"| ingest | {_m(rt.ingest_seconds, '', 1)} |",
        f"| dataset | {_m(rt.dataset_seconds, '', 1)} |",
        f"| training | {_m(rt.training_seconds, '', 1)} |",
        f"| depth render | {_m(rt.depth_render_seconds, '', 1)} |",
        f"| surface | {_m(rt.surface_build_seconds, '', 1)} |",
        f"| evaluation | {_m(rt.evaluation_seconds, '', 1)} |",
        f"| **total (this workflow)** | {_m(rt.total_seconds, '', 1)} |",
        "",
        "## Stages",
        "",
        "| stage | status | seconds | commit |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['stage']} | {row['status']} | {_m(row['elapsed_seconds'], '', 1)} | "
        f"`{(row['git_commit'] or '—')[:12]}` |"
        for row in r.stages
    ]
    lines += [
        "",
        "---",
        "",
        "Reused stages contributed no time to this workflow and are marked `reused`. A value "
        "shown as — was not available; it is `null` in `phase2_report.json`, which is the "
        "machine-readable source of truth this document renders.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: Phase2Report, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = report.save(out / REPORT_JSON_FILE)
    md_path = out / REPORT_MD_FILE
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def report_from_workflow(work_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Regenerate the report from artifacts that already exist. Runs no stage.

    This is the second half of the CLI's split: producing the workflow and producing the
    document are different jobs, and rewriting a heading must never cost an E57 extraction or a
    training run.
    """
    from minegs.e2e.models import WORKFLOW_STATE_FILE

    work = Path(work_dir)
    state_path = work / WORKFLOW_STATE_FILE
    if not state_path.is_file():
        raise ContractError(f"{state_path}: no workflow there to report on")
    # A workflow that stopped part-way is reported, not refused: a partial report is exactly
    # what an operator wants to look at after a failure, and `_maturity` names every stage that
    # did not complete rather than leaving the reader to infer it from a blank table.
    report = build_report(WorkflowState.load(state_path))
    return write_report(report, out_dir or (work / "report"))[0]
