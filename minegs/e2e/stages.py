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

__all__ = ["STAGING_ARTIFACTS", "dataset_identity", "dataset_spec", "ingest_spec", "staging_digest"]

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

    result = build_dataset(staging, dataset_dir, spec, config_path=build_config)
    out = Path(result.dataset_dir)

    # The same three checks the operator runs by hand, in the same order, against the same
    # functions. Being the thing that built the dataset is not a reason to skip them.
    problems = validate_layout(out, result.manifest)
    if problems:
        raise ContractError(f"{out}: the dataset this stage built does not validate: {problems}")
    judgement = judge(result.manifest)
    gate = run_golden_gate(out, staging, ctx.work_dir / "golden_gate", raise_on_fail=True)

    ident = dataset_identity(out)
    report = result.report
    return StageOutcome(
        outputs={
            **ident,
            "dataset_dir": str(out.resolve()),
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
            "golden_gate_dir": str((ctx.work_dir / "golden_gate").resolve()),
        },
        command={
            "call": "minegs.dataset.materialize.build_dataset",
            "staging": str(staging),
            "out": str(dataset_dir),
            "config": str(build_config),
        },
    )


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
