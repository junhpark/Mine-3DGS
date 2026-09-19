"""Cutting a section *artifact* — the series plus what it was cut from (§11, Phase 1C)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError
from minegs.eval.sections.models import SectionRecord, SectionSource, reference_axis_of
from minegs.eval.sections.sections import extract_sections

__all__ = [
    "build_section_record",
    "reproducibility_refusal",
    "require_reproducible_sections",
    "section_source",
]

#: Relative slack when re-cutting a series and comparing it to the recorded one. The cut is
#: deterministic given the same points, axis and parameters, so this absorbs float
#: representation between builds and nothing else. It is not a tolerance on geometry.
_RTOL = 1e-9


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


def require_reproducible_sections(rec: SectionRecord, manifest, centerline) -> None:
    """``reproducibility_refusal`` as an assertion, for callers that are not the CLI."""
    reason = reproducibility_refusal(rec, manifest, centerline)
    if reason is not None:
        raise ContractError(reason)


def reproducibility_refusal(rec: SectionRecord, manifest, centerline) -> str | None:
    """Why these sections cannot carry a claim, or ``None`` if they can. Claim path only.

    Everything else in ``check_section_record`` ties the record to the right dataset, the right
    axis and the right surface. None of it says the *areas* came from that surface: the series
    lives inside the record, so editing ``area_m2`` in the JSON — or flipping an invalid station
    to a plausible number and closing a gap with it — leaves every identity check satisfied.

    A claim is not a declaration, so here the numbers are re-derived. The surface points are
    verified against their own record (``check_surface``), hopped into TLS_GLOBAL and cut again
    with the parameters the record names; the result must be the series the record carries.

    Only on the claim path. It costs a full re-extraction, and a diagnostic number is allowed to
    be its producer's word — that is most of what "diagnostic" means. And only while the surface
    is still there: a claim whose evidence has been archived is a claim nothing can check, which
    is a refusal rather than something to wave through.
    """
    from minegs.eval.surface.models import check_surface, find_surface, load_surface

    src = rec.source
    if src.kind != "surface" or not src.point_path:
        return f"sections {rec.section_id} were cut from {src.kind}, which cannot carry a claim"
    path = Path(src.point_path)
    if find_surface(path) is None:
        return (
            f"sections {rec.section_id} name surface {src.surface_id} at {path}, and it is not "
            "there. A volume_accuracy claim re-cuts the sections from the surface and compares "
            "them, so without the surface these areas are only this file's own word for them. "
            "Restore the surface artifact, or pass --diagnostic for non-claim numbers."
        )
    surface, points = load_surface(path)
    pc = check_surface(surface, points, rec.dataset_id, rec.dataset_hash)
    if pc.frame != "TLS_GLOBAL":
        pc = pc.transformed(manifest.T_tls_from_local, "TLS_GLOBAL")
    p = rec.parameters
    again = extract_sections(
        pc.xyz,
        centerline,
        interval_m=float(rec.series.interval_m),
        thickness_m=float(rec.series.thickness_m),
        angle_bins=int(rec.series.angle_bins),
        start_m=p.get("start_m"),
        end_m=p.get("end_m"),
        frame="TLS_GLOBAL",
    )
    return _compare_series(rec, again)


def _compare_series(rec: SectionRecord, again) -> str | None:
    where = f"sections {rec.section_id}"
    if len(again.sections) != len(rec.series.sections):
        return (
            f"{where}: re-cutting surface {rec.source.surface_id} gives {len(again.sections)} "
            f"stations, the record holds {len(rec.series.sections)}"
        )
    valid_now = [s.valid for s in again.sections]
    valid_rec = [s.valid for s in rec.series.sections]
    if valid_now != valid_rec:
        flipped = [
            f"{s.chainage_m:g}"
            for s, a, b in zip(rec.series.sections, valid_rec, valid_now, strict=True)
            if a != b
        ]
        return (
            f"{where}: re-cutting surface {rec.source.surface_id} disagrees about which stations "
            f"are observed at {flipped[:8]}. A station the surface does not support cannot be "
            "made observed by editing the series."
        )
    for name, got, want in (
        ("area_m2", again.areas(), rec.series.areas()),
        ("radii_m", _radii(again), _radii(rec.series)),
    ):
        if not np.allclose(got, want, rtol=_RTOL, atol=0.0, equal_nan=True):
            bad = int(np.argmax(~np.isclose(got, want, rtol=_RTOL, atol=0.0, equal_nan=True)))
            return (
                f"{where}: re-cutting surface {rec.source.surface_id} does not reproduce this "
                f"series' {name} (first disagreement at index {bad}: recorded "
                f"{np.ravel(want)[bad]!r}, re-cut {np.ravel(got)[bad]!r}). These areas were not "
                "measured from that surface."
            )
    return None


def _radii(series) -> np.ndarray:
    return np.array(
        [[np.nan if v is None else v for v in s.radii_m] for s in series.sections],
        dtype=np.float64,
    )
