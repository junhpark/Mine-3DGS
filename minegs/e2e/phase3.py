"""The image/360 survey, carried by the same workflow (§Phase 3 C4).

Phase 2's workflow takes an E57 to a report. Phase 3 asks the same question of a survey that
has no scanner in it: a video or a set of panoramas, reconstructed by SfM, brought into the
survey frame by a *measured* similarity, and evaluated by the same gates. What changes is the
first two stages and nothing after them.

That is the whole design here. ``INGEST`` still means "turn the raw capture into geometry in
the survey frame" — for images that is the frame set, the reconstruction and the registration,
because a reconstruction that has not been registered is not in any frame anyone can evaluate
against. ``DATASET`` still means "build the dataset and check it, with the checks the operator
would run by hand". ``TRAIN`` through ``REPORT`` are the Phase 2 handlers, untouched: one
implementation of training, depth, surface, geometry, sections and volume, exercised by both
paths. A second copy of them would be a second answer waiting to disagree with the first.

**What this path may never do.** The TLS reference enters as evaluation evidence and nowhere
else: it is the ICP target when the operator asked for one, the cloud the golden gate projects
into the views, and the held-out reference the volume is measured against. It is never the
initialisation — ``build_dataset_from_sfm`` re-derives that the init cloud is the
reconstruction's own points rather than taking the manifest's word — and whatever the
registration was fitted against is recorded, so an overlap between that support and the
evaluation holdout is a refusal rather than a small number (§Phase 3 AD-2).

**Substitution.** Two more seams than Phase 2 has, for the same reason: this machine has no
COLMAP and no ffmpeg. A substituted SfM still has to leave a model ``check_sfm`` accepts and a
record that says ``real_sfm_execution = false``; a substituted extractor still has to leave
frames the real selection, crop, mask and digest checks accept. Both are recorded on the stage
and both reach the report. **A workflow with either substituted is not G2**, and no number it
produces is evidence about a mine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file, sha256_tree
from minegs.e2e.models import Stage
from minegs.e2e.runner import StageContext, StageOutcome, StageSpec
from minegs.e2e.stages import (
    dataset_identity,
    default_specs,
    depth_spec,
    fresh_artifact_dir,
    geometry_spec,
    report_spec,
    sections_volume_spec,
    surface_spec,
    train_spec,
)

__all__ = [
    "phase3_specs",
    "sfm_dataset_spec",
    "video_ingest_spec",
]

#: Everything under a supplied image directory. A frame that was added, removed or repainted
#: between two executions is a different capture, and the fingerprint has to say so.
_IMAGE_TREE_PATTERNS = ("**/*",)


def _capture_identity(p3: Any) -> dict[str, Any]:
    """What was captured, by digest. One of a video file or a directory of frames."""
    if bool(p3.video) == bool(p3.image_dir):
        raise ContractError(
            "the phase3 block must name exactly one capture: `video` (frames are extracted "
            "from it) or `image_dir` (frames were extracted elsewhere). Naming both leaves it "
            "undecided which one the reconstruction is of; naming neither leaves nothing to "
            "reconstruct."
        )
    if p3.video:
        src = Path(p3.video)
        if not src.is_file():
            raise ContractError(f"{src}: no such video file")
        return {"capture": "video", "name": src.name, "sha256": sha256_file(src)}
    src = Path(p3.image_dir)
    if not src.is_dir():
        raise ContractError(f"{src}: no such frame directory")
    return {
        "capture": "image_dir",
        "name": src.name,
        "sha256": sha256_tree(src, _IMAGE_TREE_PATTERNS),
    }


def _model_json(value: Any) -> Any:
    """A settings object as plain JSON, whether it is a pydantic model or a dataclass.

    ``PanoConvention`` is a frozen dataclass and the ring spec is a model; both end up in a
    fingerprint, so both have to serialise the same way rather than one of them raising.
    """
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return asdict(value)
    return value


# ---------------------------------------------------------------- INGEST (video -> registered)


def _video_ingest_inputs(ctx: StageContext) -> dict[str, Any]:
    """The capture's identity and every option that decides what comes out of it.

    All three sub-steps are fingerprinted together because they are one statement: these
    frames, reconstructed this way, put into the survey frame by this measurement. Changing
    the blur threshold changes which images SfM saw, and changing the targets changes where the
    result ended up — neither leaves the registered reconstruction the same thing.
    """
    p3 = ctx.config.require_phase3()
    ident = _capture_identity(p3)
    return {
        "source_capture": ident["capture"],
        "source_name": ident["name"],
        "source_sha256": ident["sha256"],
        "kind": p3.kind,
        "selection": {
            "fps": p3.fps,
            "scale_width": p3.scale_width,
            "start_s": p3.start_s,
            "duration_s": p3.duration_s,
            "frame_pattern": p3.frame_pattern,
            "blur_threshold": p3.blur_threshold,
            "hamming_threshold": p3.hamming_threshold,
            "max_frames": p3.max_frames,
        },
        "ring": _model_json(p3.ring),
        "pano_convention": _model_json(p3.pano_convention),
        "nadir_el_deg": p3.nadir_el_deg,
        "mask_boxes": [list(b) for b in (p3.mask_boxes or [])],
        "sfm": {
            "backend": p3.sfm_backend,
            "mapper": p3.sfm_mapper,
            "options": _model_json(p3.sfm_options),
            "component": p3.sfm_component,
        },
        "registration": {
            "basis": p3.basis,
            "targets_sfm_sha256": _digest(p3.targets_sfm, "targets file"),
            "targets_tls_sha256": _digest(p3.targets_tls, "targets file"),
            "target_ranges_m": _ranges(p3.target_ranges_m),
            "icp_target_sha256": _digest(p3.icp_target_ply, "ICP target"),
            "icp_ranges_m": _ranges(p3.icp_ranges_m),
            "icp_target_is_whole_reference": p3.icp_target_is_whole_reference,
            "icp_max_dist_m": p3.icp_max_dist_m,
            "icp_iters": p3.icp_iters,
            "inlier_m": p3.inlier_m,
            "thresholds": dict(p3.registration_thresholds or {}) or None,
            "reference_sha256": _digest(ctx.config.tls_reference_ply, "TLS reference"),
        },
    }


def _digest(path: str | None, what: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"{p}: no such {what}")
    return sha256_file(p)


def _ranges(value: list[tuple[float, float]] | None) -> list[list[float]] | None:
    return None if value is None else [[float(lo), float(hi)] for lo, hi in value]


def _video_ingest_run(ctx: StageContext) -> StageOutcome:
    from minegs.eval.register.run import register_sfm
    from minegs.ingest.video.build import build_frameset
    from minegs.ingest.video.sfm.run import run_sfm

    p3 = ctx.config.require_phase3()
    fs_target = Path(p3.frameset_dir) if p3.frameset_dir else ctx.work_dir / "artifacts/frameset"
    sfm_target = Path(p3.sfm_dir) if p3.sfm_dir else ctx.work_dir / "artifacts/sfm"
    reg_target = (
        Path(p3.registration_dir)
        if p3.registration_dir
        else ctx.work_dir / "artifacts/registration"
    )

    rec_fs, fs_dir = build_frameset(
        fs_target,
        kind=p3.kind,
        video=p3.video,
        image_dir=p3.image_dir,
        fps=p3.fps,
        scale_width=p3.scale_width,
        start_s=p3.start_s,
        duration_s=p3.duration_s,
        pattern=p3.frame_pattern,
        blur_threshold=p3.blur_threshold,
        hamming_threshold=p3.hamming_threshold,
        max_frames=p3.max_frames,
        ring=p3.ring,
        pano_convention=p3.pano_convention,
        nadir_el_deg=p3.nadir_el_deg,
        mask_boxes=p3.mask_boxes,
        overwrite=ctx.rebuilding,
        extractor=ctx.frame_extractor,
    )
    rec_sfm, sfm_dir = run_sfm(
        fs_dir,
        sfm_target,
        backend=p3.sfm_backend,
        mapper=p3.sfm_mapper,
        options=p3.sfm_options,
        component=p3.sfm_component,
        executor=ctx.sfm,
        overwrite=ctx.rebuilding,
    )
    rec_reg, reg_dir = register_sfm(
        sfm_dir,
        reg_target,
        basis=p3.basis,
        targets_sfm=p3.targets_sfm,
        targets_tls=p3.targets_tls,
        target_ranges_m=p3.target_ranges_m,
        icp_target_ply=p3.icp_target_ply,
        icp_ranges_m=p3.icp_ranges_m,
        icp_target_is_whole_reference=p3.icp_target_is_whole_reference,
        reference_ply=ctx.config.tls_reference_ply,
        icp_max_dist_m=p3.icp_max_dist_m,
        icp_iters=p3.icp_iters,
        inlier_m=p3.inlier_m,
        thresholds=p3.registration_thresholds,
        overwrite=ctx.rebuilding,
    )
    ident = _capture_identity(p3)
    return StageOutcome(
        outputs={
            "mode": "image_reconstruction",
            # The keys the report's source summary reads. A video survey has an identity too,
            # and leaving these blank would have the report say the survey is unidentified.
            "source_file_name": ident["name"],
            "source_sha256": ident["sha256"],
            "source_capture": ident["capture"],
            "output_frame": rec_reg.target_frame,
            "frameset_id": rec_fs.frameset_id,
            "frameset_dir": str(fs_dir.resolve()),
            "frameset_kind": rec_fs.kind,
            "images_sha256": rec_fs.images_sha256,
            "n_frames_considered": rec_fs.selection.considered,
            "n_frames_kept": rec_fs.selection.kept,
            "n_frames_rejected": rec_fs.selection.considered - rec_fs.selection.kept,
            "n_images": len(rec_fs.images),
            "n_crops": len(rec_fs.crops),
            "n_masks": len(rec_fs.masks),
            "frame_extraction_method": rec_fs.extraction.method,
            "frame_extraction_real": rec_fs.extraction.real_execution,
            "sfm_id": rec_sfm.sfm_id,
            "sfm_dir": str(sfm_dir.resolve()),
            "sfm_backend": rec_sfm.backend,
            "sfm_model_sha256": rec_sfm.model_sha256,
            "sfm_registered_images": rec_sfm.registered_images,
            "sfm_points": rec_sfm.points,
            "sfm_selected_component": rec_sfm.selected_component,
            "sfm_component_count": len(rec_sfm.components),
            "sfm_metric_state": rec_sfm.metric_state,
            "real_sfm_execution": rec_sfm.real_sfm_execution,
            "registration_id": rec_reg.registration_id,
            "registration_dir": str(reg_dir.resolve()),
            "registration_sha256": sha256_file(reg_dir / "registration.json"),
            "registration_basis": rec_reg.basis,
            "registration_scale": float(rec_reg.scale),
            "registration_rmse_m": float(rec_reg.diagnostics.rmse_m),
            "registration_inlier_ratio": float(rec_reg.diagnostics.inlier_ratio),
            "registration_support_ranges_m": (
                None
                if rec_reg.support_ranges_m is None
                else [list(r) for r in rec_reg.support_ranges_m]
            ),
            "registration_claim_allowed": rec_reg.claim_allowed,
            "registration_claim_refusals": list(rec_reg.claim_refusals),
        },
        command={
            "call": "minegs.e2e.phase3.video_ingest_spec",
            "frameset": str(fs_target),
            "sfm": str(sfm_target),
            "registration": str(reg_target),
            "substituted_sfm": ctx.sfm is not None,
            "substituted_frame_extraction": ctx.frame_extractor is not None,
            "overwrite": ctx.rebuilding,
        },
    )


def video_ingest_spec() -> StageSpec:
    return StageSpec(stage=Stage.INGEST, inputs=_video_ingest_inputs, run=_video_ingest_run)


# ---------------------------------------------------------------- DATASET (from the SfM)


def _sfm_dataset_inputs(ctx: StageContext) -> dict[str, Any]:
    """The three artifacts the dataset is built from, re-read from disk.

    Not copied out of the ingest record: a frame set whose images changed after the SfM ran, or
    a registration edited after it was written, is a different input to this build whatever the
    ledger says. The digests below are recomputed the way the builder's own checks recompute
    them, so a build that would now be refused comes out stale here instead.
    """
    from minegs.ingest.video.models import load_frameset
    from minegs.ingest.video.sfm.models import load_sfm, model_digest

    up = ctx.upstream(Stage.INGEST)
    p3 = ctx.config.require_phase3()
    rec_fs, fs_dir = load_frameset(up["frameset_dir"])
    rec_sfm, sfm_root = load_sfm(up["sfm_dir"])
    reg_json = Path(up["registration_dir"]) / "registration.json"
    if not reg_json.is_file():
        raise ContractError(
            f"{reg_json}: the registration this dataset would be built from is gone"
        )
    return {
        "frameset_id": rec_fs.frameset_id,
        # The tree, not the record's own ``images_sha256``: that field is a statement the frame
        # set made about bytes, and asking the record whether those bytes are still themselves
        # can only ever get the answer yes.
        "frameset_tree_sha256": sha256_tree(fs_dir, _IMAGE_TREE_PATTERNS),
        "sfm_id": rec_sfm.sfm_id,
        "sfm_model_sha256": model_digest(sfm_root / rec_sfm.model_dir),
        "registration_sha256": sha256_file(reg_json),
        "dataset_config": p3.dataset.model_dump(mode="json"),
        "centerline_sha256": _digest(p3.dataset.centerline_file, "centerline CSV"),
        "tls_reference_sha256": _digest(ctx.config.tls_reference_ply, "TLS reference"),
    }


def _sfm_dataset_run(ctx: StageContext) -> StageOutcome:
    from minegs.core.manifest import validate_layout
    from minegs.dataset.from_sfm import build_dataset_from_sfm
    from minegs.dataset.golden_gate_sfm import run_image_only_gate
    from minegs.eval.protocol import judge

    cfg = ctx.config
    p3 = cfg.require_phase3()
    up = ctx.upstream(Stage.INGEST)
    (dataset_dir,) = cfg.require("dataset_dir")

    manifest, ds = build_dataset_from_sfm(
        up["frameset_dir"],
        up["sfm_dir"],
        up["registration_dir"],
        dataset_dir,
        p3.dataset,
        overwrite=ctx.rebuilding,
    )

    # The same three checks the operator runs by hand, against the same functions. Having built
    # the dataset is not a reason to skip them.
    problems = validate_layout(ds, manifest)
    if problems:
        raise ContractError(f"{ds}: the dataset this stage built does not validate: {problems}")
    judgement = judge(manifest)
    gate_dir = fresh_artifact_dir(ctx, "golden_gate")
    gate = run_image_only_gate(
        ds, gate_dir, reference_ply=cfg.tls_reference_ply, raise_on_fail=True
    )

    extent = _centerline_extent(ds, manifest)
    reg = manifest.registration
    return StageOutcome(
        outputs={
            **dataset_identity(ds),
            "dataset_dir": str(Path(ds).resolve()),
            "centerline_extent_m": list(extent) if extent else None,
            "centerline_length_m": (float(extent[1]) - float(extent[0])) if extent else None,
            "n_stations": len(manifest.capture_groups),
            "n_images": len(manifest.all_images()),
            "init_points": manifest.initialization.n_points,
            "init_source": manifest.initialization.source,
            "camera_convention": {},
            "protocols": [p.value for p in judgement.protocols],
            "claims": [c.value for c in judgement.claims],
            "geometry_holdout_ranges_m": [list(r) for r in judgement.holdout_ranges_m],
            "refusals": list(judgement.refusals),
            # Named so no reader can take this for the station-scan gate. The image-only gate
            # answers a different question with different evidence (§Phase 3 C3).
            "golden_gate_kind": gate["gate_kind"],
            "golden_gate_result": gate["structural_result"],
            "golden_gate_passed": gate["structural_result"] == "pass",
            "golden_gate_problems": list(gate["problems"]),
            "golden_gate_dir": str(gate_dir.resolve()),
            "real_data_validation_status": gate["real_data_validation_status"],
            "registration_claim_allowed": None if reg is None else reg.claim_allowed,
            "registration_support_ranges_m": None if reg is None else reg.support_ranges_m,
            "scale_basis": manifest.scale.basis if manifest.scale else None,
            "scale_factor": manifest.scale.factor if manifest.scale else None,
        },
        command={
            "call": "minegs.dataset.from_sfm.build_dataset_from_sfm",
            "frameset": up["frameset_dir"],
            "sfm": up["sfm_dir"],
            "registration": up["registration_dir"],
            "out": str(dataset_dir),
            "overwrite": ctx.rebuilding,
        },
    )


def _centerline_extent(dataset_dir: Path, manifest: Any) -> tuple[float, float] | None:
    from minegs.e2e.stages import _centerline_extent as extent

    return extent(Path(dataset_dir), manifest)


def sfm_dataset_spec() -> StageSpec:
    return StageSpec(stage=Stage.DATASET, inputs=_sfm_dataset_inputs, run=_sfm_dataset_run)


# ---------------------------------------------------------------- the set


def phase3_specs() -> dict[Stage, StageSpec]:
    """The image/360 path: two replaced stages, and Phase 2's for everything after them."""
    specs = dict(default_specs())
    specs[Stage.INGEST] = video_ingest_spec()
    specs[Stage.DATASET] = sfm_dataset_spec()
    # Named rather than left implicit, so a stage added to Phase 2 and not considered here
    # shows up as a difference rather than as silent inheritance.
    for spec in (
        train_spec,
        depth_spec,
        surface_spec,
        geometry_spec,
        sections_volume_spec,
        report_spec,
    ):
        s = spec()
        specs[s.stage] = s
    return specs
