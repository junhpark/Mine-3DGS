"""The stage handlers: what each step of the survey calls, and what it records (§Phase 2).

Every handler is a call into an existing domain module and a description of what came out. None
of them reimplements anything — ``minegs ingest e57 extract`` and the INGEST stage run the same
``extract()``, ``minegs dataset from-e57`` and the DATASET stage run the same ``build_dataset()``
— so there is one implementation of each step in the repository and the orchestrated path
cannot drift from the hand-run one.

Two rules hold throughout, and they are what keep the orchestrator out of the evidence chain.

**A stage's ``inputs`` reads the world, not the ledger.** It locates things through the previous
stage's outputs and then re-derives their identity from disk: the dataset hash is recomputed
from the dataset directory, the surface digest from the surface file. A stage whose recorded
inputs came from a world that has since changed is therefore stale on its own terms, and a
dataset edited after it was built invalidates training without anyone having to notice.

**A stage does not vouch for what it produced.** Having just built a surface is not evidence
that the surface verifies; the handler calls the same validator the CLI calls and lets it
refuse. "We made it here, so it is trusted" is the one sentence an orchestrator must never be
allowed to say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minegs.core.errors import ContractError
from minegs.core.provenance import sha256_file, sha256_tree
from minegs.e2e.models import Stage
from minegs.e2e.runner import StageContext, StageOutcome, StageSpec
from minegs.eval.surface.models import DEPTH_MANIFEST_FILE

__all__ = [
    "STAGING_ARTIFACTS",
    "dataset_identity",
    "dataset_spec",
    "default_specs",
    "depth_spec",
    "geometry_spec",
    "ingest_spec",
    "report_spec",
    "sections_volume_spec",
    "staging_digest",
    "surface_spec",
    "train_spec",
]

#: The three files that make a staging tree what it is (``minegs/dataset/staging_input.py``).
STAGING_ARTIFACTS = ("inventory.json", "pano_mapping.json", "extraction_manifest.json")


def staging_digest(staging: str | Path) -> str:
    """Identity of an extraction. The manifest inside records a digest per scan and image."""
    root = Path(staging)
    missing = [n for n in STAGING_ARTIFACTS if not (root / n).is_file()]
    if missing:
        raise ContractError(
            f"{root}: not a staging tree — missing {missing}. The input to a dataset build is a "
            "Phase 0B.3 staging tree written by `minegs ingest e57 extract`."
        )
    return sha256_tree(root, STAGING_ARTIFACTS)


def dataset_identity(dataset_dir: str | Path) -> dict[str, str]:
    """``dataset_id`` and the hash *as the dataset is now* — never as a record remembers it."""
    from minegs.core.manifest import Manifest
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    d = Path(dataset_dir)
    manifest = Manifest.load_dataset(d, strict_layout=False)
    return {
        "dataset_id": manifest.dataset_id,
        "dataset_hash": sha256_tree(d, DATASET_HASH_PATTERNS),
    }


def fresh_artifact_dir(ctx: StageContext, name: str) -> Path:
    """A never-before-used directory under the workflow, because artifacts are written once.

    Every publish in this project refuses an existing target (``staged_dir``,
    ``build_depth_surface``, ``refuse_used_run_dir``), which is what stops an interrupted write
    from being read as a thinner artifact. A fixed path per stage would make ``--rebuild-from``
    collide with the run it is replacing, so each execution gets its own and the ledger records
    which one it used.
    """
    from minegs.core.provenance import make_id

    return ctx.work_dir / "artifacts" / name / make_id(name)


def _digest_if_there(path: str | Path | None) -> str | None:
    p = Path(path) if path else None
    return sha256_file(p) if p and p.is_file() else None


# ---------------------------------------------------------------- INGEST


def _ingest_inputs(ctx: StageContext) -> dict[str, Any]:
    """The survey's identity, and the options that decide what comes out of it.

    The E57 is digested rather than stamped: size and mtime say a file was replaced, not that it
    is a different survey, and this digest is what every downstream stage ultimately hangs off.
    On a multi-gigabyte scan that costs a pass over the file each time the workflow is
    consulted, which is the price of the guarantee (docs/PHASE2_E57_G2.md).
    """
    cfg = ctx.config
    if not cfg.source_e57:
        # Adoption: an extraction performed elsewhere. Its identity is the staging tree itself.
        (staging,) = cfg.require("staging_dir")
        return {"mode": "adopted_staging", "staging_digest": staging_digest(staging)}
    src = Path(cfg.source_e57)
    if not src.is_file():
        raise ContractError(f"{src}: no such E57 file")
    return {
        "mode": "extract",
        "source_e57": src.name,
        "source_sha256": sha256_file(src),
        "voxel_m": cfg.voxel_m,
        "max_scan_points": cfg.max_scan_points,
        "scan_ids": sorted(cfg.scan_ids),
        "mapping_sha256": _digest_if_there(cfg.mapping),
        "vendor_manifest_sha256": _digest_if_there(cfg.vendor_manifest),
        "images_dir": cfg.images_dir,
    }


def _ingest_run(ctx: StageContext) -> StageOutcome:
    from minegs.ingest.e57.extract import extract
    from minegs.ingest.e57.inventory import inventory

    cfg = ctx.config
    (staging,) = cfg.require("staging_dir")
    if not cfg.source_e57:
        return _adopt_staging(staging)

    src = Path(cfg.source_e57)
    inv = inventory(src, compute_hash=True)
    manifest = extract(
        src,
        staging,
        scan_ids=list(cfg.scan_ids) or None,
        voxel_m=cfg.voxel_m,
        registered=True,
        with_images=True,
        mapping=cfg.mapping,
        vendor_manifest=cfg.vendor_manifest,
        images_dir=cfg.images_dir,
        compute_hash=True,
        # Only when `--rebuild-from ingest` asked for it. The extractor refuses an occupied
        # staging tree because two runs' scans cannot be told apart afterwards, and that
        # refusal is right on every run except this one -- which is why the flag exists.
        overwrite=ctx.rebuilding,
        max_scan_points=cfg.max_scan_points,
    )
    if manifest.source_sha256 and inv.file.sha256 and manifest.source_sha256 != inv.file.sha256:
        # Two independent passes over the same file. Disagreeing means it changed under us.
        raise ContractError(
            f"{src} hashed to {inv.file.sha256[:12]} when inventoried and "
            f"{manifest.source_sha256[:12]} when extracted; the file changed during ingest"
        )
    mapping_report = manifest.mapping_report
    return StageOutcome(
        outputs={
            "mode": "extract",
            "source_e57": str(src.resolve()),
            "source_file_name": src.name,
            "source_sha256": manifest.source_sha256 or inv.file.sha256,
            "source_size_bytes": inv.file.size_bytes,
            "e57_writer": inv.file.e57_library_version,
            "coordinate_metadata": inv.file.coordinate_metadata,
            "scan_count_declared": inv.scan_count,
            "staging_dir": str(Path(staging).resolve()),
            "staging_digest": staging_digest(staging),
            "registration": manifest.registration,
            "output_frame": manifest.output_frame,
            "n_scans_extracted": len(manifest.scan_outputs),
            "n_images_extracted": len(manifest.image_outputs),
            "n_images_skipped": len(manifest.skipped_images),
            "mapping_status_counts": _mapping_counts(mapping_report),
            "issues": list(manifest.issues),
        },
        command={
            "call": "minegs.ingest.e57.extract.extract",
            "source_e57": str(src),
            "work_dir": str(staging),
            "voxel_m": cfg.voxel_m,
            "max_scan_points": cfg.max_scan_points,
            "scan_ids": list(cfg.scan_ids),
            "mapping": cfg.mapping,
            "vendor_manifest": cfg.vendor_manifest,
            "images_dir": cfg.images_dir,
            "overwrite": ctx.rebuilding,
        },
    )


def _adopt_staging(staging: str | Path) -> StageOutcome:
    """Record an extraction this workflow did not perform, and say that it did not.

    Legitimate — a survey extracted on the machine that had the disk for it — and recorded
    distinctly, because a report that cannot tell an adopted staging tree from one it produced
    cannot say what this workflow is evidence of.
    """
    from minegs.dataset.staging_input import load_staging

    tree = load_staging(staging)
    ex = tree.extraction
    return StageOutcome(
        outputs={
            "mode": "adopted_staging",
            "source_e57": ex.source_e57,
            "source_file_name": Path(ex.source_e57).name,
            "source_sha256": ex.source_sha256,
            "source_size_bytes": tree.inventory.file.size_bytes,
            "e57_writer": tree.inventory.file.e57_library_version,
            "coordinate_metadata": tree.inventory.file.coordinate_metadata,
            "scan_count_declared": tree.inventory.scan_count,
            "staging_dir": str(Path(staging).resolve()),
            "staging_digest": staging_digest(staging),
            "registration": ex.registration,
            "output_frame": ex.output_frame,
            "n_scans_extracted": len(ex.scan_outputs),
            "n_images_extracted": len(ex.image_outputs),
            "n_images_skipped": len(ex.skipped_images),
            "mapping_status_counts": _mapping_counts(ex.mapping_report),
            "issues": list(ex.issues),
            "adopted": True,
        },
        command={"call": "minegs.dataset.staging_input.load_staging", "staging": str(staging)},
    )


def _mapping_counts(report: Any) -> dict[str, int]:
    """How each image got its station, preserved rather than summarised into a pass/fail.

    Phase 0B.2 refuses to infer a mapping from index or file order, and a workflow that reported
    only "mapped: 12" would lose the distinction between evidence and assumption.
    """
    if report is None:
        return {}
    counts: dict[str, int] = {}
    for rec in getattr(report, "mappings", []) or []:
        counts[rec.status] = counts.get(rec.status, 0) + 1
    return counts


def ingest_spec() -> StageSpec:
    return StageSpec(stage=Stage.INGEST, inputs=_ingest_inputs, run=_ingest_run)


# ---------------------------------------------------------------- DATASET


def _dataset_inputs(ctx: StageContext) -> dict[str, Any]:
    up = ctx.upstream(Stage.INGEST)
    (build_config,) = ctx.config.require("build_config")
    cfgp = Path(build_config)
    if not cfgp.is_file():
        raise ContractError(f"{cfgp}: build config not found")
    return {
        # Re-read from disk, not copied from the ingest record: a staging tree edited after
        # extraction is a different input to this build, whatever the ledger says.
        "staging_digest": staging_digest(up["staging_dir"]),
        "build_config_sha256": sha256_file(cfgp),
    }


def _dataset_run(ctx: StageContext) -> StageOutcome:
    from minegs.core.manifest import validate_layout
    from minegs.dataset.build_config import load_build_config
    from minegs.dataset.calibrate import calibrate_camera_convention
    from minegs.dataset.golden_gate import run_golden_gate
    from minegs.dataset.materialize import build_dataset
    from minegs.dataset.staging_input import load_staging
    from minegs.eval.protocol import judge

    cfg = ctx.config
    staging = ctx.upstream(Stage.INGEST)["staging_dir"]
    build_config, dataset_dir = cfg.require("build_config", "dataset_dir")
    spec = load_build_config(build_config)

    calibration = _ensure_camera_convention(
        spec, staging, calibrate_camera_convention, load_staging
    )

    # Same rule as ingest: an existing dataset is refused unless this execution's
    # `--rebuild-from dataset` said to replace it. The builder's own foreign-file and
    # ownership checks still run, so this asks it to replace what it made, not to delete a
    # directory blind.
    result = build_dataset(
        staging, dataset_dir, spec, config_path=build_config, overwrite=ctx.rebuilding
    )
    out = Path(result.dataset_dir)

    # The same three checks the operator runs by hand, in the same order, against the same
    # functions. Being the thing that built the dataset is not a reason to skip them.
    problems = validate_layout(out, result.manifest)
    if problems:
        raise ContractError(f"{out}: the dataset this stage built does not validate: {problems}")
    judgement = judge(result.manifest)
    gate_dir = fresh_artifact_dir(ctx, "golden_gate")
    gate = run_golden_gate(out, staging, gate_dir, raise_on_fail=True)

    ident = dataset_identity(out)
    report = result.report
    # Read back from the dataset's own centerline file rather than measured over some other
    # polyline: the report states how long the tunnel this dataset describes is, and it has to
    # be the same axis the holdout ranges were cut out of.
    extent = _centerline_extent(out, result.manifest)
    return StageOutcome(
        outputs={
            **ident,
            "dataset_dir": str(out.resolve()),
            "centerline_extent_m": list(extent) if extent else None,
            "centerline_length_m": (float(extent[1]) - float(extent[0])) if extent else None,
            "n_stations": report.get("n_stations"),
            "n_images": report.get("n_images"),
            "init_points": (report.get("initialization") or {}).get("points_out"),
            "camera_convention": report.get("camera_convention") or {},
            "camera_convention_sha256": _digest_if_there(spec.camera.convention_file),
            "camera_convention_status": calibration,
            "protocols": [p.value for p in judgement.protocols],
            "claims": [c.value for c in judgement.claims],
            "geometry_holdout_ranges_m": [list(r) for r in judgement.holdout_ranges_m],
            "refusals": list(judgement.refusals),
            "golden_gate_result": gate.get("structural_result"),
            "golden_gate_passed": gate.get("structural_result") == "pass",
            "golden_gate_problems": list(gate.get("structural_problems") or []),
            "roundtrip_error_m": gate.get("roundtrip_error_m"),
            "golden_gate_dir": str(gate_dir.resolve()),
        },
        command={
            "call": "minegs.dataset.materialize.build_dataset",
            "staging": str(staging),
            "out": str(dataset_dir),
            "config": str(build_config),
            "overwrite": ctx.rebuilding,
        },
    )


def _centerline_extent(dataset_dir: Path, manifest: Any) -> tuple[float, float] | None:
    """The chainage this dataset's centerline spans, or None when it declares none."""
    from minegs.core.centerline import Centerline

    ref = getattr(manifest, "centerline", None)
    if ref is None:
        return None
    cl = Centerline.from_csv(dataset_dir / ref.file, ref.frame, ref.source)
    return (cl.s_start, cl.s_end)


def _ensure_camera_convention(spec: Any, staging: str | Path, calibrate: Any, load: Any) -> str:
    """Measure the axis convention when the build config names a file that is not there yet.

    Never hard-coded and never defaulted: the convention is evidence, produced by projecting a
    station's own scan into its own images (§12). An existing file is the operator's evidence
    and is left exactly as it is — silently recalculating over it would replace a reviewed
    measurement with an unreviewed one.
    """
    camera = getattr(spec, "camera", None)
    target = getattr(camera, "convention_file", None) if camera else None
    if not target:
        return "not_required"
    path = Path(target)
    if path.is_file():
        return "supplied"
    result = calibrate(load(staging))
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    return f"measured:{result.status}"


def dataset_spec() -> StageSpec:
    return StageSpec(stage=Stage.DATASET, inputs=_dataset_inputs, run=_dataset_run)


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- TRAIN


def _train_inputs(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.config
    dataset_dir = ctx.upstream(Stage.DATASET)["dataset_dir"]
    return {
        **dataset_identity(dataset_dir),
        "profile": cfg.profile,
        "backend": cfg.backend,
        "runner": cfg.runner,
        "native": cfg.native,
    }


def _train_run(ctx: StageContext) -> StageOutcome:
    from minegs.eval.surface.depth import check_run
    from minegs.train.runner.base import RunnerConfig
    from minegs.train.runner.local import RunConfig

    cfg = ctx.config
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    ident = dataset_identity(dataset_dir)
    run_cfg = RunConfig(
        dataset_dir=str(dataset_dir),
        profile=cfg.profile,
        backend=cfg.backend,
        runner=cfg.runner,
    )
    from minegs.core.provenance import make_id

    base = Path(cfg.runs_dir) if cfg.runs_dir else ctx.work_dir / "artifacts" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_cfg.run_dir = str(base / make_id("run"))
    runner_cfg = RunnerConfig(runner=cfg.runner, native=cfg.native)

    trainer = ctx.trainer or _default_trainer
    run_dir = Path(trainer(run_cfg, runner_cfg))

    # The same check `eval surface-depth` makes before it will touch a run: succeeded, this
    # dataset, this hash, LOCAL_METRIC outputs. Having just launched the run is not evidence
    # that it finished one, and a trainer injected for a test is not evidence of anything.
    record = check_run(run_dir, ident["dataset_id"], ident["dataset_hash"])
    checkpoint = run_dir / record.final_checkpoint if record.final_checkpoint else None
    return StageOutcome(
        outputs={
            "run_id": record.run_id,
            "run_dir": str(run_dir.resolve()),
            "status": record.status.value,
            "backend": dict(record.backend or {}),
            "profile_name": cfg.profile,
            "runner": record.runner,
            "checkpoint_step": record.checkpoint_step,
            "final_checkpoint": record.final_checkpoint,
            "checkpoint_sha256": _digest_if_there(checkpoint),
            "staged": dict(record.staged or {}),
            "frame_of_outputs": record.frame_of_outputs,
            "real_gpu_execution": _real_gpu_execution(record, substituted=ctx.trainer is not None),
            **dataset_identity(dataset_dir),
        },
        command={
            "call": "minegs.train.runner.local.LocalRunner.submit",
            "dataset_dir": str(dataset_dir),
            "profile": cfg.profile,
            "backend": cfg.backend,
            "runner": cfg.runner,
            "native": cfg.native,
            "substituted_trainer": ctx.trainer is not None,
        },
        runtime_env=_training_env(record, substituted=ctx.trainer is not None),
    )


def _default_trainer(run_cfg: Any, runner_cfg: Any) -> Path:
    """Submit and block. The CLI's own path, including its refusal to detach."""
    from minegs.train.runner import get_runner
    from minegs.train.runner.base import RunStatus, load_record

    handle = get_runner(runner_cfg.runner, runner_cfg).submit(run_cfg)
    status = handle.wait(poll_s=2.0)
    if status is not RunStatus.SUCCEEDED:
        raise ContractError(
            load_record(handle.run_dir).failure_reason or f"run finished {status.value}"
        )
    return Path(handle.run_dir)


def _real_gpu_execution(record: Any, substituted: bool) -> bool:
    """Whether a real GPU trained this run, decided by what the run recorded about the machine.

    The question being asked was the weaker one -- "was the test seam left alone" -- which is
    nearly right on the local path, since `LocalRunner` refuses to start without CUDA. But the
    report's claim is about hardware, and a claim about hardware is answered by the hardware's
    own evidence (§0D.2 D2-2), not by which code path this process happened to take. A
    substituted trainer is False whatever the host has; a real one with nothing recorded about
    a GPU is False too, because nothing establishes that one was there.
    """
    if substituted:
        return False
    evidence = dict(getattr(record, "runtime", None) or {})
    return bool(evidence.get("gpu_model")) or evidence.get("torch_cuda_available") is True


def _training_env(record: Any, substituted: bool) -> dict[str, Any]:
    """What the machine was, as the run recorded it — and null with a reason when it did not.

    A substituted trainer never ran on a GPU, so nothing here is filled in from the host: the
    host having a GPU would say nothing about where these weights came from.
    """
    if substituted:
        return {
            "gpu_model": None,
            "cuda_version": None,
            "reason": "the training backend was substituted; no GPU executed this run",
        }
    evidence = dict(getattr(record, "runtime", None) or {})
    env: dict[str, Any] = {
        "gpu_model": evidence.get("gpu_model"),
        "cuda_version": evidence.get("torch_cuda") or evidence.get("cuda"),
        "driver_version": evidence.get("driver_version"),
        "torch": evidence.get("torch"),
        "gsplat": evidence.get("gsplat"),
        "source": evidence.get("source"),
    }
    if not env["gpu_model"]:
        env["reason"] = "the run record carries no GPU identification"
    return env


def train_spec() -> StageSpec:
    return StageSpec(stage=Stage.TRAIN, inputs=_train_inputs, run=_train_run)


# ---------------------------------------------------------------- DEPTH


def _depth_inputs(ctx: StageContext) -> dict[str, Any]:
    up = ctx.upstream(Stage.TRAIN)
    run_dir = Path(up["run_dir"])
    from minegs.train.runner.base import load_record

    record = load_record(run_dir)
    checkpoint = run_dir / record.final_checkpoint if record.final_checkpoint else None
    return {
        "run_id": record.run_id,
        "run_status": record.status.value,
        "checkpoint": record.final_checkpoint,
        "checkpoint_sha256": _digest_if_there(checkpoint),
        **dataset_identity(ctx.upstream(Stage.DATASET)["dataset_dir"]),
        "min_alpha": ctx.config.min_alpha,
    }


def _depth_run(ctx: StageContext) -> StageOutcome:
    from minegs.eval.surface.render import render_depths

    cfg = ctx.config
    run_dir = Path(ctx.upstream(Stage.TRAIN)["run_dir"])
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    out = fresh_artifact_dir(ctx, "depth")
    manifest, depth_dir = render_depths(
        run_dir, dataset_dir, out, cfg.min_alpha, renderer=ctx.renderer
    )
    ratios = [d.valid_ratio for d in manifest.depths]
    return StageOutcome(
        outputs={
            "depth_manifest_id": manifest.manifest_id,
            "depth_dir": str(Path(depth_dir).resolve()),
            "depth_manifest_sha256": _digest_if_there(Path(depth_dir) / DEPTH_MANIFEST_FILE),
            "renderer": (manifest.renderer or {}).get("name"),
            "renderer_version": (manifest.renderer or {}).get("version"),
            "depth_map_count": len(manifest.depths),
            "mean_valid_ratio": sum(ratios) / len(ratios) if ratios else None,
            "run_id": manifest.run_id,
            "real_renderer_execution": ctx.renderer is None,
        },
        command={
            "call": "minegs.eval.surface.render.render_depths",
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "out": str(out),
            "min_alpha": cfg.min_alpha,
            "substituted_renderer": ctx.renderer is not None,
        },
    )


def depth_spec() -> StageSpec:
    return StageSpec(stage=Stage.DEPTH, inputs=_depth_inputs, run=_depth_run)


# ---------------------------------------------------------------- SURFACE


def _surface_inputs(ctx: StageContext) -> dict[str, Any]:
    """The depth this surface is fused from — the maps themselves, not just their manifest.

    Hashing ``depth_manifest.json`` says which maps were declared; it says nothing about what
    is in them. A ``.npy`` edited in place left the manifest byte-identical, so the surface
    stayed fresh, and the geometry and volume built on it stayed fresh with it. ``depth_digest``
    is the same order-independent digest ``build_depth_surface`` records in the artifact, so
    what the fingerprint covers is exactly what the surface consumed.
    """
    from minegs.eval.surface.depth import depth_digest
    from minegs.ingest.common.colmap_io import read_model

    up = ctx.upstream(Stage.DEPTH)
    depth_dir = Path(up["depth_dir"])
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    model = read_model(dataset_dir / "sparse" / "0")
    return {
        "depth_manifest_id": up["depth_manifest_id"],
        "depth_manifest_sha256": _digest_if_there(depth_dir / DEPTH_MANIFEST_FILE),
        "depth_maps_sha256": depth_digest(depth_dir, model.images),
        "run_id": ctx.upstream(Stage.TRAIN)["run_id"],
        **dataset_identity(dataset_dir),
        "stride": ctx.config.stride,
        "max_depth_m": ctx.config.max_depth_m,
    }


def _surface_run(ctx: StageContext) -> StageOutcome:
    from minegs.eval.surface.depth import build_depth_surface, rederive_depth_source
    from minegs.eval.surface.models import check_surface, load_surface

    cfg = ctx.config
    depth_dir = Path(ctx.upstream(Stage.DEPTH)["depth_dir"])
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    run_dir = Path(ctx.upstream(Stage.TRAIN)["run_dir"])
    out = fresh_artifact_dir(ctx, "surface")
    _built, surface_dir = build_depth_surface(
        depth_dir, dataset_dir, run_dir, out, cfg.stride, cfg.max_depth_m
    )

    # Re-read it from disk through the same validators an evaluation would, and re-derive the
    # promotion rather than trusting the value this stage just wrote. The orchestrator having
    # produced an artifact is not evidence about the artifact (§Phase 1C).
    ident = dataset_identity(dataset_dir)
    surface, points = load_surface(surface_dir)
    cloud = check_surface(surface, points, ident["dataset_id"], ident["dataset_hash"])
    stale = rederive_depth_source(surface, dataset_dir)
    if stale is not None:
        raise ContractError(f"the surface this stage built does not promote: {stale}")
    return StageOutcome(
        outputs={
            "surface_id": surface.surface_id,
            "surface_dir": str(Path(surface_dir).resolve()),
            "point_sha256": surface.point_sha256,
            "point_count": surface.point_count,
            "depth_map_count": surface.depth_map_count,
            "depth_source": surface.depth_source,
            "run_id": surface.run_id,
            "span_m": list(surface.span_m or []),
            "verified_point_count": len(cloud),
            **ident,
        },
        command={
            "call": "minegs.eval.surface.depth.build_depth_surface",
            "depth_dir": str(depth_dir),
            "dataset_dir": str(dataset_dir),
            "run_dir": str(run_dir),
            "out": str(out),
            "stride": cfg.stride,
            "max_depth": cfg.max_depth_m,
        },
    )


def surface_spec() -> StageSpec:
    return StageSpec(stage=Stage.SURFACE, inputs=_surface_inputs, run=_surface_run)


# ---------------------------------------------------------------- GEOMETRY


def _evaluation_inputs(ctx: StageContext) -> dict[str, Any]:
    """What both evaluation stages hang off: which surface, which dataset, which reference.

    The surface is read back off disk and verified, not copied out of the ledger. Copying is
    what a fingerprint exists to avoid: the recorded ``point_sha256`` is a statement the
    SURFACE stage made about a file, and asking the ledger whether that file is still itself
    can only ever get the answer yes. Replacing the published PLY therefore left every
    evaluation fresh, and the geometry and volume numbers already in the report went on
    describing a surface that was no longer there.

    ``check_surface`` is the same gate ``minegs eval geometry`` applies, so a file that no
    longer matches its record raises here and the stages that rest on it come out stale. The
    depth those points were fused from reaches this through the chained upstream fingerprint,
    which now covers the maps themselves.
    """
    from minegs.eval.surface.models import check_surface, load_surface

    surface_dir = Path(ctx.upstream(Stage.SURFACE)["surface_dir"])
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    (tls,) = ctx.config.require("tls_reference_ply")
    ref = Path(tls)
    if not ref.is_file():
        raise ContractError(f"{ref}: no TLS reference cloud there")
    ident = dataset_identity(dataset_dir)
    record, points = load_surface(surface_dir)
    check_surface(record, points, ident["dataset_id"], ident["dataset_hash"])
    return {
        "surface_id": record.surface_id,
        "point_sha256": record.point_sha256,
        # Recorded here so an edited verdict moves the fingerprint too. Whether it still holds
        # is re-derived rather than read, by the claim gate these stages run (§1C).
        "depth_source": record.depth_source,
        "surface_run_id": record.run_id,
        **ident,
        "tls_reference_sha256": sha256_file(ref),
    }


def _geometry_inputs(ctx: StageContext) -> dict[str, Any]:
    return {**_evaluation_inputs(ctx), "max_dist_m": ctx.config.max_dist_m}


def _geometry_run(ctx: StageContext) -> StageOutcome:
    from minegs.eval.geometry.evaluate import evaluate_geometry

    cfg = ctx.config
    surface_dir = ctx.upstream(Stage.SURFACE)["surface_dir"]
    dataset_dir = ctx.upstream(Stage.DATASET)["dataset_dir"]
    (tls,) = cfg.require("tls_reference_ply")
    # holdout_only and diagnostic are not options here. Phase 2 is the claim-bearing path, and
    # the flags that downgrade it exist for an operator who has decided to look at something
    # else; a workflow that quietly chose the downgraded variant would report a weaker number
    # under a stronger heading.
    result = evaluate_geometry(surface_dir, dataset_dir, tls, True, cfg.max_dist_m, False)
    rep = result.report
    out = fresh_artifact_dir(ctx, "geometry")
    write_json(out / "geometry.json", rep.model_dump(mode="json"))
    return StageOutcome(
        outputs={
            "claim": rep.claim,
            "chainage_range_m": list(rep.chainage_range_m) if rep.chainage_range_m else None,
            "max_dist_m": rep.max_dist_m,
            "accuracy_median_m": rep.accuracy.median_m,
            "accuracy_p95_m": rep.accuracy.p95_m,
            "completeness_median_m": rep.completeness.median_m,
            "completeness_p95_m": rep.completeness.p95_m,
            "chamfer_m": rep.chamfer_m,
            "f_score": dict(rep.f_score),
            "report_path": str((out / "geometry.json").resolve()),
            "notes": list(result.notes),
        },
        command={
            "call": "minegs.eval.geometry.evaluate.evaluate_geometry",
            "pred": str(surface_dir),
            "dataset_dir": str(dataset_dir),
            "tls_ply": str(tls),
            "holdout_only": True,
            "max_dist_m": cfg.max_dist_m,
            "diagnostic": False,
        },
    )


def geometry_spec() -> StageSpec:
    return StageSpec(stage=Stage.GEOMETRY, inputs=_geometry_inputs, run=_geometry_run)


# ---------------------------------------------------------------- SECTIONS_VOLUME


def _sections_inputs(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.config
    return {
        **_evaluation_inputs(ctx),
        "interval_m": cfg.interval_m,
        "thickness_m": cfg.thickness_m,
        "angle_bins": cfg.angle_bins,
        "holdout_ranges_m": ctx.upstream(Stage.DATASET)["geometry_holdout_ranges_m"],
    }


def _sections_run(ctx: StageContext) -> StageOutcome:
    from minegs.core.pointcloud import read_ply
    from minegs.eval.geometry.evaluate import (
        load_dataset_and_centerline,
        resolve_prediction,
        to_tls,
    )
    from minegs.eval.protocol import judge
    from minegs.eval.sections import (
        build_section_record,
        check_claim_evidence,
        check_section_record,
        section_source,
    )
    from minegs.eval.volume import compare_to_reference, integrate_sections, plan_integration

    cfg = ctx.config
    surface_dir = Path(ctx.upstream(Stage.SURFACE)["surface_dir"])
    dataset_dir = Path(ctx.upstream(Stage.DATASET)["dataset_dir"])
    (tls,) = cfg.require("tls_reference_ply")
    manifest, centerline = load_dataset_and_centerline(dataset_dir)
    judgement = judge(manifest)
    ranges = [tuple(r) for r in judgement.holdout_ranges_m]
    if not ranges:
        raise ContractError(
            f"{manifest.dataset_id} declares no geometry holdout, so there is no held-out TLS to "
            "validate the reconstruction against (§5)"
        )

    cut = {
        "interval_m": cfg.interval_m,
        "thickness_m": cfg.thickness_m,
        "angle_bins": cfg.angle_bins,
    }
    # The prediction goes through the claim-path resolver: a surface that does not verify, or
    # whose promotion does not re-derive, stops here rather than producing a comparison.
    resolved = resolve_prediction(surface_dir, dataset_dir, manifest, diagnostic=False)
    pred_points = to_tls(resolved.points, manifest)
    pred = build_section_record(
        pred_points.xyz,
        section_source(resolved.surface, surface_dir),
        dataset_dir,
        manifest,
        centerline,
        **cut,
    )
    # The reference is the held-out TLS. It is a raw cloud and is recorded as one: it is the
    # thing being compared *against*, and dressing it up as a reconstruction would make the
    # section artifact say something false about where it came from.
    ref_cloud = read_ply(tls)
    if ref_cloud.frame != "TLS_GLOBAL":
        raise ContractError(
            f"{tls}: the TLS reference is in frame {ref_cloud.frame}, not TLS_GLOBAL; sections "
            "of it would be cut along chainages of a different coordinate system"
        )
    ref = build_section_record(
        ref_cloud.xyz, section_source(None, tls), dataset_dir, manifest, centerline, **cut
    )

    out = fresh_artifact_dir(ctx, "sections_volume")
    pred_path, ref_path = out / "sections_predicted.json", out / "sections_reference.json"
    pred.save(pred_path)
    ref.save(ref_path)
    for record in (pred, ref):
        check_section_record(
            record,
            manifest.dataset_id,
            dataset_identity(dataset_dir)["dataset_hash"],
            dataset_dir,
            manifest,
            centerline,
        )

    validation = compare_to_reference(pred, ref, ranges)
    paired_path = write_json(out / "paired_validation.json", validation.model_dump(mode="json"))

    # What the prediction alone would integrate over the holdout, and whether it could carry a
    # volume_accuracy claim. Reported, not re-decided: `minegs eval volume` owns that gate and
    # these are the inputs it reads (§Phase 1C).
    predicted = integrate_sections(pred.series, pred.reference_axis, ranges=ranges)
    write_json(out / "volume_predicted.json", predicted.model_dump(mode="json"))
    evidence = check_claim_evidence(pred, manifest, centerline, dataset_dir, ranges)
    coverage_complete = plan_integration(pred.series, ranges)[1].complete

    v = validation.volume
    return StageOutcome(
        outputs={
            "section_id_predicted": pred.section_id,
            "section_id_reference": ref.section_id,
            "sections_predicted_path": str(pred_path.resolve()),
            "sections_reference_path": str(ref_path.resolve()),
            "paired_validation_path": str(paired_path.resolve()),
            # The report reads this file rather than the ledger, so the ledger has to be able
            # to say whether it is still the file this stage wrote (§P2 C5).
            "paired_validation_sha256": sha256_file(paired_path),
            "volume_predicted_path": str((out / "volume_predicted.json").resolve()),
            "grid": dict(validation.grid),
            "requested_ranges_m": [list(r) for r in ranges],
            "paired_valid_count": validation.sections.paired_valid_count,
            "section_mae_m2": validation.sections.mean_absolute_error_m2,
            "predicted_volume_m3": v.predicted_volume_m3,
            "reference_volume_m3": v.reference_volume_m3,
            "absolute_error_m3": v.absolute_error_m3,
            "coverage_fraction": v.coverage_fraction,
            "predicted_only_volume_m3": predicted.volume_m3,
            "predicted_only_coverage_fraction": predicted.coverage.coverage_fraction,
            "volume_accuracy_available": evidence.refusal is None and coverage_complete,
            "volume_accuracy_refusal": evidence.refusal
            or (None if coverage_complete else "holdout coverage is incomplete"),
            "max_point_gap_m": evidence.max_point_gap_m,
        },
        command={
            "call": "minegs.eval.volume.paired.compare_to_reference",
            "surface_dir": str(surface_dir),
            "dataset_dir": str(dataset_dir),
            "tls_reference_ply": str(tls),
            **cut,
            "ranges": [list(r) for r in ranges],
        },
    )


def sections_volume_spec() -> StageSpec:
    return StageSpec(stage=Stage.SECTIONS_VOLUME, inputs=_sections_inputs, run=_sections_run)


# ---------------------------------------------------------------- REPORT


def _report_inputs(ctx: StageContext) -> dict[str, Any]:
    """The artifact the report actually reads, by digest.

    Everything else it needs comes from the ledger, and the ledger's own identities arrive
    through the chained upstream fingerprint. The paired validation is the one file it opens, so
    it is the one thing that can move under it without any earlier stage noticing.
    """
    sv = ctx.upstream(Stage.SECTIONS_VOLUME)
    paired = Path(sv["paired_validation_path"])
    if not paired.is_file():
        raise ContractError(
            f"{paired}: the paired validation the sections/volume stage recorded is not there, "
            "so there is nothing to report on. Re-run that stage (`--rebuild-from "
            "sections_volume`)."
        )
    return {
        "paired_validation_sha256": sha256_file(paired),
        # Named, not read: it is what the report is a report *of*, and a workflow whose geometry
        # artifact was replaced is not the workflow this document describes.
        "geometry_report_sha256": _digest_if_there(ctx.upstream(Stage.GEOMETRY).get("report_path")),
    }


def _report_run(ctx: StageContext) -> StageOutcome:
    """Aggregate. No science: every number is copied from an artifact that verified itself.

    The stage row for REPORT itself reads ``running`` in the document, because that is what it
    is doing while it writes it. Regenerating afterwards with ``report_from_workflow`` shows the
    finished status instead; neither version invents one.
    """
    from minegs.e2e.report import build_report, write_report

    out = fresh_artifact_dir(ctx, "report")
    report = build_report(ctx.state)
    json_path, md_path = write_report(report, out)
    m = report.maturity
    return StageOutcome(
        outputs={
            "report_id": report.report_id,
            "report_json_path": str(json_path.resolve()),
            "report_md_path": str(md_path.resolve()),
            "structural_status": m.structural_status,
            "real_data_validation_status": m.real_data_validation_status,
            "human_visual_review_status": m.human_visual_review_status,
            "real_gpu_execution": report.training.real_gpu_execution,
            "real_renderer_execution": report.reconstruction.real_renderer_execution,
        },
        command={"call": "minegs.e2e.report.build_report", "out": str(out)},
    )


def _report_finalise(ctx: StageContext, outcome: StageOutcome) -> None:
    """Write the report again, now that the ledger this report describes is finished.

    The first write happens inside the stage, where the stage can only describe itself as
    ``running``; the final ledger says ``succeeded``, and a reader comparing the two found the
    workflow's own account of itself disagreeing with the workflow. Same id, same directory,
    same inputs -- only the row that could not be known yet changes.
    """
    from minegs.e2e.report import build_report, write_report

    write_report(
        build_report(ctx.state, report_id=outcome.outputs["report_id"]),
        Path(outcome.outputs["report_json_path"]).parent,
    )


def report_spec() -> StageSpec:
    return StageSpec(
        stage=Stage.REPORT, inputs=_report_inputs, run=_report_run, finalise=_report_finalise
    )


def default_specs() -> dict[Stage, StageSpec]:
    """The whole chain, in order. One place, so the CLI and a test drive the same stages."""
    return {
        Stage.INGEST: ingest_spec(),
        Stage.DATASET: dataset_spec(),
        Stage.TRAIN: train_spec(),
        Stage.DEPTH: depth_spec(),
        Stage.SURFACE: surface_spec(),
        Stage.GEOMETRY: geometry_spec(),
        Stage.SECTIONS_VOLUME: sections_volume_spec(),
        Stage.REPORT: report_spec(),
    }
