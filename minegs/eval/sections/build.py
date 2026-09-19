"""Cutting a section *artifact* — the series plus what it was cut from (§11, Phase 1C)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError
from minegs.eval.sections.models import SectionRecord, SectionSource, reference_axis_of
from minegs.eval.sections.sections import extract_sections

__all__ = ["build_section_record", "section_source"]


def section_source(surface, path: str | Path) -> SectionSource:
    """Describe the cloud the sections are about to be cut from.

    *surface* is the ``SurfaceRecord`` when one was resolved, ``None`` for a raw PLY. The
    distinction is the whole point of the record, so it is derived from what was actually
    loaded — never passed in by a caller who could name the value it wants.
    """
    from minegs.core.provenance import sha256_file

    path = Path(path)
    if surface is None:
        if not path.is_file():
            raise ContractError(f"{path}: not a surface artifact and not a PLY file")
        return SectionSource(kind="raw_cloud", point_sha256=sha256_file(path), point_path=str(path))
    return SectionSource(
        kind="surface",
        surface_id=surface.surface_id,
        run_id=surface.run_id,
        depth_source=surface.depth_source,
        point_sha256=surface.point_sha256,
        point_path=str(path),
    )


def build_section_record(
    points_tls: np.ndarray,
    source: SectionSource,
    dataset_dir: str | Path,
    manifest,
    centerline,
    *,
    interval_m: float = 1.0,
    thickness_m: float = 0.1,
    angle_bins: int = 180,
    start_m: float | None = None,
    end_m: float | None = None,
) -> SectionRecord:
    """Cut sections in TLS_GLOBAL and wrap them in the provenance a volume claim needs."""
    from minegs.core.provenance import make_id, sha256_file, sha256_tree, stamp
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    dataset_dir = Path(dataset_dir)
    if interval_m <= 0:
        raise ContractError(f"--interval-m must be > 0, got {interval_m}")
    if thickness_m <= 0:
        raise ContractError(f"--thickness-m must be > 0, got {thickness_m}")
    if angle_bins < 3:
        raise ContractError(f"--angle-bins must be >= 3 to bound an area, got {angle_bins}")
    if centerline.frame != "TLS_GLOBAL":
        raise ContractError(
            f"the reference axis is in frame {centerline.frame}; sections are cut in TLS_GLOBAL"
        )

    series = extract_sections(
        points_tls,
        centerline,
        interval_m=interval_m,
        thickness_m=thickness_m,
        angle_bins=angle_bins,
        start_m=start_m,
        end_m=end_m,
        frame="TLS_GLOBAL",
    )
    if not series.sections:
        raise ContractError(
            f"no stations between {start_m} and {end_m} on a reference axis spanning "
            f"{centerline.s_start:g}-{centerline.s_end:g} m; there is nothing to section"
        )
    parameters: dict[str, Any] = {
        "interval_m": float(interval_m),
        "thickness_m": float(thickness_m),
        "angle_bins": int(angle_bins),
        "start_m": None if start_m is None else float(start_m),
        "end_m": None if end_m is None else float(end_m),
        "point_count": len(points_tls),
    }
    axis_file = dataset_dir / manifest.centerline.file
    if not axis_file.is_file():
        raise ContractError(f"{axis_file}: the dataset's reference axis file is missing")
    return SectionRecord(
        section_id=make_id("sections"),
        dataset_id=manifest.dataset_id,
        dataset_hash=sha256_tree(dataset_dir, DATASET_HASH_PATTERNS),
        source=source,
        reference_axis=reference_axis_of(manifest),
        reference_axis_sha256=sha256_file(axis_file),
        series=series,
        parameters=parameters,
        provenance=stamp(
            parameters,
            parents=[p for p in (manifest.dataset_id, source.surface_id, source.run_id) if p],
        ),
    )
