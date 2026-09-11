"""E57 inventory — read what the file *declares*, without interpreting it (Phase 0B.1).

Scope and non-scope, per docs/ROADMAP.md:

* This module opens an E57, enumerates its Data3D scans, and records what each one declares:
  identity, point count, which point fields exist, pose, header bounds. It also notes whether
  image structures are present.
* It does **not** read point data. ``header.point_count`` is ``points.childCount()`` and the
  bounds come from header metadata, so *metadata parsing* is O(scan count), not O(points),
  and uses no meaningful memory (§17). Note that the default SHA-256 for provenance does read
  the whole file byte by byte, so wall-clock is O(file size) unless ``compute_hash=False``.
* It does **not** extract, decode or map panoramas. Detection only; the station↔panorama
  contract is Phase 0B.2.
* It does **not** declare a coordinate system. An E57's own coordinates are ``SOURCE``.

Every optional field is accessed through a guard, because E57 files differ by vendor and
export workflow: a missing name, guid, pose or bounds is recorded as absent, never guessed.
Notably ``pye57``'s own ``ScanHeader.rotation_matrix`` / ``.translation`` fall back to identity
and zeros when a pose is absent, so this module reads the pose node itself instead (§10: no
silent identity fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.provenance import (
    ProvenanceRecord,
    SourceAsset,
    git_commit,
    sha256_file,
    tool_versions,
)
from minegs.ingest.e57 import _nodes
from minegs.ingest.e57.exceptions import (
    E57NoScansError,
    E57UnsupportedStructureError,
)
from minegs.ingest.e57.models import (
    E57FileInfo,
    E57Inventory,
    E57ScanInventory,
    ImageSummary,
    PoseStatus,
    ScanBounds,
    ScanPose,
    StationCandidate,
    make_bounds,
    make_scan_pose,
    scan_id_for,
    station_id_for,
)

# E57 standard point-field names -> what minegs can rely on.
CARTESIAN_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")
SPHERICAL_FIELDS = ("sphericalRange", "sphericalAzimuth", "sphericalElevation")
RGB_FIELDS = ("colorRed", "colorGreen", "colorBlue")
ROW_COLUMN_FIELDS = ("rowIndex", "columnIndex")

#: Header fields worth keeping for diagnosis. Read only if the scan declares them.
_VENDOR_STRING_FIELDS = ("description", "sensorVendor", "sensorModel", "sensorSerialNumber")
_VENDOR_SCALAR_FIELDS = ("temperature", "relativeHumidity", "atmosphericPressure")


# Guarded node access lives in one module so every E57 reader shares the same guards; see
# ``_nodes`` for why. These aliases keep this file readable.
_fields = _nodes.fields
_value = _nodes.value
_str_value = _nodes.str_value


# ---------------------------------------------------------------- scan-level readers


def _pose_from_header(
    header: Any, scan_fields: list[str]
) -> tuple[ScanPose | None, PoseStatus, list[str], list[str]]:
    """Read the scan's pose node directly. Returns (pose or None, status, issues, notes).

    The status distinguishes the three ways a pose can fail to be usable. In particular a
    declared-but-unreadable pose is ``unreadable``, never ``absent``: the pose object is
    ``None`` in both cases, so without the status a parse failure would be indistinguishable
    from an honestly unregistered scan and would count as a usable scan.
    """
    issues: list[str] = []
    notes: list[str] = []
    if "pose" not in scan_fields:
        return None, "absent", issues, notes
    try:
        pose_node = header["pose"]
        pose_fields = _fields(pose_node)
    except Exception as e:
        return None, "unreadable", [f"pose node declared but unreadable: {e}"], notes
    if not pose_fields:
        return (
            None,
            "unreadable",
            ["pose node declared but its contents could not be enumerated"],
            notes,
        )

    quat = None
    R = np.eye(3)
    if "rotation" in pose_fields:
        comps = [_value(pose_node, "rotation", k) for k in ("w", "x", "y", "z")]
        if any(c is None for c in comps):
            issues.append("pose.rotation is missing one of w/x/y/z")
        else:
            quat = np.array([float(c) for c in comps])
            R = _quat_to_rotmat_unnormalised(quat)
    else:
        issues.append("pose declares no rotation; treating rotation as unknown")

    t = np.zeros(3)
    if "translation" in pose_fields:
        comps = [_value(pose_node, "translation", k) for k in ("x", "y", "z")]
        if any(c is None for c in comps):
            issues.append("pose.translation is missing one of x/y/z")
        else:
            t = np.array([float(c) for c in comps])
    else:
        issues.append("pose declares no translation; treating translation as unknown")

    # Anything structural found above means part of the pose is unknown; hand it to the
    # validator so the pose is marked invalid rather than quietly completed with identity.
    pose = make_scan_pose(R, t, quat, extra_issues=list(issues))
    issues = list(pose.validation.issues)
    notes.extend(pose.validation.warnings)
    if not pose.validation.valid:
        status: PoseStatus = "invalid"
    elif pose.is_identity:
        status = "identity"
    else:
        status = "valid"
    return pose, status, issues, notes


def scan_pose(header: Any) -> tuple[ScanPose | None, PoseStatus]:
    """The pose of one already-opened scan header, through the Phase 0B.1 contract.

    The single entry point every caller must use. Reading ``pye57``'s ``rotation_matrix`` /
    ``translation`` directly returns identity and zeros for a scan with no pose, so a second
    reader would silently disagree with the inventory about the same file.
    """
    pose, status, _issues, _notes = _pose_from_header(header, _fields(header.node))
    return pose, status


def _quat_to_rotmat_unnormalised(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> rotation, WITHOUT normalising.

    Deliberately not normalised: a non-unit quaternion in the file is a defect the caller must
    be told about, and silently normalising it would hide exactly that (§10).
    """
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _bounds_from_header(header: Any, scan_fields: list[str]) -> tuple[ScanBounds | None, list[str]]:
    """Header-declared bounds only; never computed from points."""
    if "cartesianBounds" in scan_fields:
        vals = {
            "min_x": _value(header, "cartesianBounds", "xMinimum"),
            "max_x": _value(header, "cartesianBounds", "xMaximum"),
            "min_y": _value(header, "cartesianBounds", "yMinimum"),
            "max_y": _value(header, "cartesianBounds", "yMaximum"),
            "min_z": _value(header, "cartesianBounds", "zMinimum"),
            "max_z": _value(header, "cartesianBounds", "zMaximum"),
        }
        if all(v is not None for v in vals.values()):
            bounds = make_bounds({k: float(v) for k, v in vals.items()}, "cartesianBounds")
            if bounds is not None and not bounds.valid:
                return bounds, [
                    f"header cartesianBounds are not credible: {'; '.join(bounds.issues)}"
                ]
            return bounds, []
    return None, []


def _vendor_metadata(header: Any, scan_fields: list[str]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for key in _VENDOR_STRING_FIELDS:
        if key in scan_fields and (v := _str_value(header, key)) is not None:
            out[key] = v
    for key in _VENDOR_SCALAR_FIELDS:
        if key in scan_fields and (v := _value(header, key)) is not None:
            out[key] = v
    return out or None


def _scan_inventory(header: Any, index: int) -> E57ScanInventory:
    """Build one scan's contract from its header. No point data is read."""
    try:
        point_fields = list(header.point_fields)
    except Exception as e:
        raise E57UnsupportedStructureError(
            "<scan>", f"scan {index} has no readable point prototype: {e}"
        ) from e
    scan_fields = (
        _fields(header.node)
        if hasattr(header, "node")
        else list(getattr(header, "scan_fields", []))
    )

    issues: list[str] = []
    point_count: int | None
    try:
        point_count = int(header.point_count)
    except Exception as e:
        point_count = None
        issues.append(f"point count unavailable: {e}")

    has_cartesian = all(f in point_fields for f in CARTESIAN_FIELDS)
    has_spherical = all(f in point_fields for f in SPHERICAL_FIELDS)
    if not has_cartesian and not has_spherical:
        issues.append(
            "scan declares neither a full cartesian (x/y/z) nor a full spherical "
            f"(range/azimuth/elevation) coordinate triple; point fields: {point_fields}"
        )

    notes: list[str] = []
    pose, pose_status, pose_issues, pose_notes = _pose_from_header(header, scan_fields)
    issues.extend(pose_issues)
    notes.extend(pose_notes)
    bounds, bounds_issues = _bounds_from_header(header, scan_fields)
    issues.extend(bounds_issues)

    if pose is not None and pose.is_identity:
        notes.append(
            "pose is present but identity; the file may not carry a real registration for this scan"
        )

    return E57ScanInventory(
        scan_index=index,
        scan_id=scan_id_for(index),
        name=_str_value(header, "name") if "name" in scan_fields else None,
        guid=_str_value(header, "guid") if "guid" in scan_fields else None,
        point_count=point_count,
        has_cartesian_xyz=has_cartesian,
        has_spherical=has_spherical,
        has_rgb=all(f in point_fields for f in RGB_FIELDS),
        has_intensity="intensity" in point_fields,
        has_row_column=all(f in point_fields for f in ROW_COLUMN_FIELDS),
        pose_declared="pose" in scan_fields,
        pose_status=pose_status,
        pose=pose,
        bounds=bounds,
        raw_point_fields=point_fields,
        raw_scan_fields=scan_fields,
        vendor_metadata=_vendor_metadata(header, scan_fields),
        issues=issues,
        notes=notes,
    )


# ---------------------------------------------------------------- image detection (§13)


def _image_summary(root: Any) -> ImageSummary:
    """Whether image structures exist. Nothing is decoded or mapped here (Phase 0B.2)."""
    try:
        defined = root.isDefined("images2D")
    except Exception as e:
        return ImageSummary(
            has_images2d=False,
            image_count=None,
            enumeration_status="error",
            detection_note=f"could not determine whether images2D exists: {e}",
        )
    if not defined:
        return ImageSummary(has_images2d=False, image_count=0, enumeration_status="absent")
    try:
        count = int(root["images2D"].childCount())
    except Exception as e:
        # Present but unreadable. Reporting count=0 here would turn "unknown" into "none",
        # which is the interpretation this phase refuses to make.
        return ImageSummary(
            has_images2d=True,
            image_count=None,
            enumeration_status="error",
            detection_note=f"images2D present but not enumerable: {e}",
        )
    return ImageSummary(
        has_images2d=True,
        image_count=count,
        enumeration_status="ok",
        detection_note=(
            "images2D structure exists but is empty"
            if count == 0
            else "detected only; station/panorama mapping is Phase 0B.2"
        ),
    )


def list_images2d(path: str | Path) -> list[dict[str, Any]]:
    """Enumerate ``/images2D`` entries as plain dicts.

    Kept as the legacy shape the Phase 0A ``PanoSource`` adapters read. New code should use
    :func:`minegs.ingest.e57.images.discover_embedded_images`, which returns the Phase 0B.2
    ``ImageAsset`` contract; this wrapper exists so both cannot drift apart.
    """
    from minegs.ingest.e57.images import discover_embedded_images

    return [
        {
            "index": a.source_index,
            "guid": a.guid,
            "name": a.name,
            "associated_scan_guid": a.associated_scan_guid,
            "representation": None if a.representation == "unknown" else a.representation,
            "width": a.width,
            "height": a.height,
        }
        for a in discover_embedded_images(path)
    ]


# ---------------------------------------------------------------- public entry point


def inventory(path: str | Path, compute_hash: bool = True) -> E57Inventory:
    """Inspect an E57 file and return the Phase 0B.1 contract.

    Never reads point data: metadata parsing is O(scan count) and holds no arrays.

    ``compute_hash`` is the one part that is O(file size) — the provenance SHA-256 streams the
    whole file in 1 MB chunks (bounded memory, unbounded time on a 50 GB scan). Pass
    ``compute_hash=False`` to skip it; the report records *why* the hash is absent so it can
    never be mistaken for a hashed one.
    """
    p = Path(path)
    size = p.stat().st_size if p.is_file() else 0

    with _nodes.open_e57(p) as e57:
        root = e57.root
        try:
            scan_count = int(e57.scan_count)
        except Exception as e:
            raise E57UnsupportedStructureError(p, f"no readable /data3D vector ({e})") from e
        if scan_count == 0:
            raise E57NoScansError(p)

        file_info = E57FileInfo(
            path=str(p.resolve()),
            file_name=p.name,
            size_bytes=size,
            sha256=sha256_file(p) if compute_hash else None,
            hash_skipped_reason=None if compute_hash else "requested with --no-hash",
            e57_library_version=_str_value(root, "e57LibraryVersion"),
            format_name=_str_value(root, "formatName"),
            guid=_str_value(root, "guid"),
            coordinate_metadata=_str_value(root, "coordinateMetadata"),
        )

        scans: list[E57ScanInventory] = []
        for i in range(scan_count):
            try:
                header = e57.get_header(i)
            except Exception as e:
                raise E57UnsupportedStructureError(p, f"scan {i} header unreadable ({e})") from e
            scans.append(_scan_inventory(header, i))

        images = _image_summary(root)

    file_issues, file_notes = _file_findings(scans, images)
    stations = [
        StationCandidate(
            station_id=station_id_for(i),
            scan_ids=[s.scan_id],
            origin="e57_scan",
            mapping_status="inferred_from_scan",
            note="one station assumed per scan; unconfirmed until Phase 0B.2",
        )
        for i, s in enumerate(scans)
    ]

    return E57Inventory(
        file=file_info,
        scan_count=scan_count,
        scans=scans,
        station_candidates=stations,
        images=images,
        issues=file_issues,
        notes=file_notes,
        provenance=ProvenanceRecord(
            git_commit=git_commit(),
            source_assets=[
                SourceAsset(
                    path=str(p.resolve()),
                    sha256=file_info.sha256 or "",
                    size_bytes=size,
                )
            ],
            tool_versions=tool_versions(),
        ),
    )


def _file_findings(
    scans: list[E57ScanInventory], images: ImageSummary
) -> tuple[list[str], list[str]]:
    """File-level (problems, observations).

    Reported, never fatal: inspection capability is separate from training readiness (§10).
    """
    issues: list[str] = []
    notes: list[str] = []
    n = len(scans)
    absent = [s.scan_id for s in scans if s.pose_status == "absent"]
    if absent:
        notes.append(f"{len(absent)} of {n} scans declare no pose: {absent}")
    unreadable = [s.scan_id for s in scans if s.pose_status == "unreadable"]
    if unreadable:
        issues.append(
            f"{len(unreadable)} of {n} scans declare a pose that could not be read "
            f"(this is not the same as having no pose): {unreadable}"
        )
    bad_pose = [s.scan_id for s in scans if s.pose_status == "invalid"]
    if bad_pose:
        issues.append(f"{len(bad_pose)} of {n} scans declare an invalid pose: {bad_pose}")
    identity = [s.scan_id for s in scans if s.pose_status == "identity"]
    if n > 1 and len(identity) == n:
        notes.append(
            f"all {n} scans have an identity pose; the file appears to carry no registration"
        )
    no_coords = [s.scan_id for s in scans if not s.is_usable_for_points]
    if no_coords:
        issues.append(
            f"{len(no_coords)} of {n} scans declare no usable coordinate fields: {no_coords}"
        )
    if images.enumeration_status == "error":
        issues.append(images.detection_note or "images2D could not be enumerated")
    elif not images.has_images2d:
        notes.append("no images2D structure: panoramas, if any, are external to this file")
    elif images.image_count == 0:
        notes.append("images2D structure is present but empty")
    return issues, notes
