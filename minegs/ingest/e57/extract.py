"""E57 scan and image extraction (Phase 0B.3) — the first phase that reads point data.

What comes out of here is *staging*, not a dataset. The points are still in the E57's own
SOURCE frame (or in a scan's own SCANNER frame); no TLS_GLOBAL declaration, no LOCAL_METRIC
origin, no train/test split, no ``init_points.ply``. Those are Phase 0C, and the output
layout is deliberately not shaped like ``dataset/`` so the two cannot be confused::

    <work_dir>/
      inventory.json
      pano_mapping.json
      scans/scan_000.ply  scan_000.pose.json
      images/image_000.jpg
      extraction_manifest.json

Three things in here are easy to get quietly wrong, so each is handled explicitly:

**The invalid-state mask.** An E57 scan is a fixed-length record array: ``cartesianX`` and
``colorRed`` are parallel columns, and ``cartesianInvalidState`` marks entries whose geometry
is meaningless. Dropping the invalid entries from the coordinates and then taking the first
*N* colours — which is what the Phase 0A splitter did — shifts every colour after the first
invalid point onto the wrong point. One boolean mask is built once and applied to every
column, and any column whose length disagrees with the coordinates is a hard failure.

**Colour range.** E57 states its colour range in ``colorLimits``. It is usually 0..255 and
sometimes 0..65535, and the difference is not visible in the numbers themselves. A known
range is converted once and the conversion is recorded; an unknown range is preserved raw as
extra columns rather than guessed at.

**The pose.** ``pye57.E57.read_scan(transform=True)`` builds its transform from
``ScanHeader.rotation_matrix`` / ``.translation``, which return identity and zeros for a scan
with no pose — a silent identity fallback. This module never calls it: points come from
``read_scan_raw`` and the transform comes from the validated Phase 0B.1 ``ScanPose``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.pointcloud import PointCloud, voxel_downsample, write_ply
from minegs.core.provenance import (
    ProvenanceRecord,
    SourceAsset,
    git_commit,
    sha256_file,
    tool_versions,
)
from minegs.ingest.common.geometry import spherical_to_cart
from minegs.ingest.e57 import _nodes
from minegs.ingest.e57.exceptions import E57PoseUnusableError, E57ReadScanError
from minegs.ingest.e57.images import PANORAMA_REPRESENTATIONS, ImageAsset
from minegs.ingest.e57.inventory import (
    CARTESIAN_FIELDS,
    RGB_FIELDS,
    SPHERICAL_FIELDS,
    inventory,
    scan_pose,
)
from minegs.ingest.e57.mapping import MappingRecord, PanoMappingReport, build_mapping_report
from minegs.ingest.e57.models import E57Inventory, ScanPose

#: The E57's own global coordinates, after the scan's pose is applied. NOT TLS_GLOBAL.
SOURCE_FRAME = "SOURCE"
#: One scan's own frame, before its pose is applied. Also not TLS_GLOBAL.
SCANNER_FRAME = "SCANNER"

ExtractionFrame = Literal["SOURCE", "SCANNER"]
RegistrationStatus = Literal["registered", "unregistered"]

#: Pose states an extraction may build a transform from. ``identity`` is legitimate — the
#: file declares no displacement — and is recorded so a reader can see it was not a real one.
REGISTERABLE_POSE_STATES = ("valid", "identity")

#: Per-point columns carried through to the output, masked exactly like the coordinates.
RETAINED_ATTRIBUTES = ("intensity", "rowIndex", "columnIndex")
#: Columns consumed while building the output rather than carried through.
_CONSUMED = set(CARTESIAN_FIELDS + SPHERICAL_FIELDS + RGB_FIELDS) | {
    "cartesianInvalidState",
    "sphericalInvalidState",
}

#: Image representations Phase 0B.3 writes out. Pinhole and visual-reference images are
#: recorded and skipped: handling them means perspective geometry, which this phase does not
#: have (no crop, no cube map, no COLMAP camera, no undistortion).
SUPPORTED_IMAGE_REPRESENTATIONS = PANORAMA_REPRESENTATIONS

_BLOB_SUFFIX = {"jpegImage": ".jpg", "pngImage": ".png"}
_BLOB_MAGIC = {"jpegImage": b"\xff\xd8\xff", "pngImage": b"\x89PNG\r\n\x1a\n"}
_BLOB_FORMAT = {"jpegImage": "jpeg", "pngImage": "png"}

#: Rough bytes per point held at once: pye57's own column arrays plus ours, both float64.
_BYTES_PER_POINT_PER_FIELD = 16
#: Above this, say so. pye57 has no chunked reader, so a scan is read whole (§21).
_MEMORY_NOTE_THRESHOLD_BYTES = 2 * 1024**3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScanOutput(_Strict):
    """One written point cloud, and everything needed to know what it is."""

    scan_id: str
    scan_index: int
    path: str
    #: As the header declared it, before anything was read.
    point_count_input: int | None = None
    #: After the invalid-state mask.
    point_count_masked: int
    #: After voxel downsampling; equals ``point_count_masked`` when no voxel was requested.
    point_count_output: int
    source_frame: ExtractionFrame
    registration_status: RegistrationStatus
    pose_status: str
    pose: ScanPose | None = None
    voxel_m: float | None = None
    #: Per-point columns written alongside x/y/z.
    attributes: list[str] = Field(default_factory=list)
    #: Columns present in the file that were not carried through, and why.
    dropped_attributes: dict[str, str] = Field(default_factory=dict)
    invalid_state_field: str | None = None
    invalid_points_removed: int = 0
    color_source_range: list[float] | None = None
    color_conversion: str | None = None
    xyz_dtype: str = "f8"
    sha256: str | None = None
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ImageOutput(_Strict):
    """One image, either written out of the E57 or referenced where it already lives."""

    image_id: str
    path: str
    source: Literal["e57_embedded", "external_file"]
    #: False when the file was already on disk and is referenced rather than copied.
    extracted: bool
    representation: str
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    blob_field: str | None = None
    bytes_written: int | None = None
    mapped_scan_id: str | None = None
    mapped_station_id: str | None = None
    mapping_status: str
    mapping_evidence_type: str
    sha256: str | None = None
    issues: list[str] = Field(default_factory=list)


class SkippedImage(_Strict):
    """An image that was discovered but not written, and the reason."""

    image_id: str
    representation: str
    reason: str
    mapping_status: str


class E57ExtractionManifest(VersionedModel):
    """``minegs ingest e57 extract`` result. Staging provenance, not a dataset manifest."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    source_e57: str
    source_sha256: str | None = None
    hash_skipped_reason: str | None = None
    work_dir: str
    registration: RegistrationStatus
    output_frame: ExtractionFrame
    scan_outputs: list[ScanOutput] = Field(default_factory=list)
    image_outputs: list[ImageOutput] = Field(default_factory=list)
    skipped_images: list[SkippedImage] = Field(default_factory=list)
    mapping_report: PanoMappingReport | None = None
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    def scan_output(self, scan_id: str) -> ScanOutput:
        for s in self.scan_outputs:
            if s.scan_id == scan_id:
                return s
        raise KeyError(scan_id)


# ---------------------------------------------------------------- point payload


def _column_arrays(data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v).reshape(-1) for k, v in data.items()}


def apply_invalid_state(
    columns: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], str | None, int]:
    """Drop invalid points from *every* column at once. Returns (columns, field, removed).

    The mask is built once and applied identically to coordinates, colour, intensity,
    row/column and anything else the scan carries. Slicing one column to another's length
    instead — ``rgb[: len(xyz)]`` — silently shifts every attribute after the first invalid
    point onto the wrong point, which is invisible in a viewer and fatal for colourised
    reconstruction.

    E57 ``cartesianInvalidState``: 0 = valid, 1 = direction valid but range is not, 2 = no
    valid data. Only 0 is kept; 1 carries no usable position.
    """
    if not columns:
        return columns, None, 0
    lengths = {k: len(v) for k, v in columns.items()}
    n = max(lengths.values())
    mismatched = {k: v for k, v in lengths.items() if v != n}
    if mismatched:
        raise ContractError(
            f"E57 point columns have different lengths: {mismatched} against {n}. A scan is a "
            "fixed-length record array, so this file (or the reader) is not giving parallel "
            "columns and no mask can be applied safely."
        )
    field = next(
        (f for f in ("cartesianInvalidState", "sphericalInvalidState") if f in columns), None
    )
    if field is None:
        return columns, None, 0
    keep = np.asarray(columns[field]).reshape(-1) == 0
    removed = int(np.count_nonzero(~keep))
    return {k: v[keep] for k, v in columns.items()}, field, removed


def _colour_range(header: Any) -> tuple[list[float] | None, str | None]:
    """The colour range the scan declares, from ``colorLimits``. ``None`` means unknown."""
    lo, hi = [], []
    for channel in ("Red", "Green", "Blue"):
        a = _nodes.value(header, "colorLimits", f"color{channel}Minimum")
        b = _nodes.value(header, "colorLimits", f"color{channel}Maximum")
        if a is None or b is None:
            return None, (
                "the scan declares no colorLimits, so the colour range is unknown; values are "
                "preserved raw as color_red/green/blue columns rather than assumed to be 0..255"
            )
        lo.append(float(a))
        hi.append(float(b))
    if len(set(hi)) != 1 or len(set(lo)) != 1:
        return [min(lo), max(hi)], (
            f"colour channels declare different ranges (min {lo}, max {hi}); values are "
            "preserved raw rather than rescaled per channel"
        )
    return [lo[0], hi[0]], None


def _colour_columns(
    columns: dict[str, np.ndarray], header: Any
) -> tuple[np.ndarray | None, dict[str, np.ndarray], list[float] | None, str | None, list[str]]:
    """(rgb uint8 or None, raw colour columns to keep, declared range, conversion, issues)."""
    if not all(f in columns for f in RGB_FIELDS):
        return None, {}, None, None, []
    rng, why = _colour_range(header)
    if rng is None or why is not None:
        return (
            None,
            {f"color_{c.lower()}": columns[f"color{c}"] for c in ("Red", "Green", "Blue")},
            rng,
            why,
            [why or "colour range unknown"],
        )
    lo, hi = rng
    stacked = np.column_stack([columns[f].astype(np.float64) for f in RGB_FIELDS])
    if (lo, hi) == (0.0, 255.0):
        return stacked.astype(np.uint8), {}, rng, "none (already 0..255)", []
    if hi <= lo:
        return (
            None,
            {f"color_{c.lower()}": columns[f"color{c}"] for c in ("Red", "Green", "Blue")},
            rng,
            f"declared colour range {lo}..{hi} is empty; values preserved raw",
            [f"colour range {lo}..{hi} is not usable"],
        )
    scaled = (stacked - lo) * (255.0 / (hi - lo))
    return (
        np.clip(np.rint(scaled), 0, 255).astype(np.uint8),
        {},
        rng,
        f"linear {lo}..{hi} -> 0..255",
        [],
    )


def build_cloud(columns: dict[str, np.ndarray], header: Any) -> tuple[PointCloud, dict[str, Any]]:
    """Columns (already masked) -> a SCANNER-frame cloud plus what was done to build it."""
    if all(f in columns for f in CARTESIAN_FIELDS):
        xyz = np.column_stack([columns[f] for f in CARTESIAN_FIELDS]).astype(np.float64)
        coordinate_source = "cartesian"
    elif all(f in columns for f in SPHERICAL_FIELDS):
        xyz = spherical_to_cart(*(columns[f] for f in SPHERICAL_FIELDS))
        coordinate_source = "spherical"
    else:
        raise ContractError(
            "scan declares neither a full cartesian (x/y/z) nor a full spherical "
            f"(range/azimuth/elevation) triple; columns present: {sorted(columns)}"
        )

    rgb, raw_colour, colour_range, colour_conversion, issues = _colour_columns(columns, header)
    extra: dict[str, np.ndarray] = dict(raw_colour)
    attributes = list(raw_colour)
    for name in RETAINED_ATTRIBUTES:
        if name in columns:
            extra[name] = columns[name]
            attributes.append(name)
    dropped = {
        k: "not carried into Phase 0B staging; re-read it from the source E57 if needed"
        for k in columns
        if k not in _CONSUMED and k not in RETAINED_ATTRIBUTES
    }
    cloud = PointCloud(xyz, rgb, extra=extra, frame=SCANNER_FRAME)
    return cloud, {
        "coordinate_source": coordinate_source,
        "attributes": (["red", "green", "blue"] if rgb is not None else []) + attributes,
        "dropped_attributes": dropped,
        "color_source_range": colour_range,
        "color_conversion": colour_conversion,
        "issues": issues,
    }


def read_scan_points(
    handle: Any, index: int, header: Any | None = None
) -> tuple[PointCloud, dict[str, Any]]:
    """Read one scan's payload into a SCANNER-frame cloud. No pose is applied here.

    Uses ``read_scan_raw``, never ``read_scan``: the latter applies ``ScanHeader``'s pose
    properties, which fall back to identity for a scan with no pose.
    """
    header = header if header is not None else handle.get_header(index)
    try:
        raw = handle.read_scan_raw(index)
    except AssertionError:
        raise  # a test guard saying this reader should not have touched the payload
    except Exception as e:
        raise E57ReadScanError(index, str(e)) from e
    columns, field, removed = apply_invalid_state(_column_arrays(raw))
    cloud, meta = build_cloud(columns, header)
    meta["invalid_state_field"] = field
    meta["invalid_points_removed"] = removed
    meta["point_count_masked"] = len(cloud)
    return cloud, meta


# ---------------------------------------------------------------- preflight


def _estimated_peak_bytes(point_count: int | None, field_count: int) -> int | None:
    if point_count is None:
        return None
    return point_count * max(field_count, 3) * _BYTES_PER_POINT_PER_FIELD


def plan_scans(
    inv: E57Inventory,
    scan_ids: list[str] | None = None,
    registered: bool = True,
    max_scan_points: int | None = None,
) -> tuple[list[Any], list[str]]:
    """Choose the scans to extract, refusing the whole run if any of them cannot be.

    Fail-closed and *before* anything is written: a run that writes three scans and then
    refuses the fourth leaves a staging directory that looks complete.
    """
    selected = [s for s in inv.scans if scan_ids is None or s.scan_id in scan_ids]
    if scan_ids is not None:
        missing = sorted(set(scan_ids) - {s.scan_id for s in inv.scans})
        if missing:
            raise ContractError(
                f"no such scan in this file: {missing}. Available: {[s.scan_id for s in inv.scans]}"
            )
    if not selected:
        raise ContractError("no scans selected")

    notes: list[str] = []
    broken = [(s.scan_id, s.pose_status) for s in selected if s.pose_is_broken]
    if broken:
        raise E57PoseUnusableError(
            f"{len(broken)} of {len(selected)} selected scans declare a pose that cannot be "
            f"used: {', '.join(f'{i} ({st})' for i, st in broken)}. A declared-but-broken pose "
            "is not the same as no pose, and neither may be silently replaced by identity. "
            "Run `minegs ingest e57 inventory` to see why"
        )
    if registered:
        unregistered = [(s.scan_id, s.pose_status) for s in selected if s.pose_status == "absent"]
        if unregistered:
            raise E57PoseUnusableError(
                f"{len(unregistered)} of {len(selected)} selected scans declare no pose: "
                f"{', '.join(f'{i} ({st})' for i, st in unregistered)}. Registered extraction "
                "places points in the file's SOURCE frame and needs one. Use --raw to write "
                "scanner-frame clouds instead, which are marked unregistered and are not "
                "interchangeable with registered output"
            )
    unusable = [s.scan_id for s in selected if not s.is_usable_for_points]
    if unusable:
        raise ContractError(
            f"{len(unusable)} of {len(selected)} selected scans declare no usable coordinate "
            f"fields: {unusable}"
        )

    for s in selected:
        peak = _estimated_peak_bytes(s.point_count, len(s.raw_point_fields))
        if peak is None:
            continue
        if max_scan_points is not None and (s.point_count or 0) > max_scan_points:
            raise ContractError(
                f"{s.scan_id} declares {s.point_count:,} points, above the --max-scan-points "
                f"limit of {max_scan_points:,}. pye57 reads a scan whole, so this would need "
                f"roughly {peak / 1024**3:.1f} GB of RAM. Raise the limit, or use the PDAL "
                "tiling path (`minegs ingest e57 tiles`) for production-scale clouds"
            )
        if peak > _MEMORY_NOTE_THRESHOLD_BYTES:
            notes.append(
                f"{s.scan_id} declares {s.point_count:,} points; pye57 has no chunked reader, "
                f"so extracting it holds roughly {peak / 1024**3:.1f} GB at once. Use "
                "--max-scan-points to fail closed instead, or PDAL tiling for large clouds"
            )
    return selected, notes


def _prepare_work_dir(work_dir: Path, overwrite: bool) -> None:
    """Refuse an occupied directory, or empty it. Never write one run on top of another.

    ``overwrite`` *replaces*: the previous ``scans/`` and ``images/`` are removed first.
    Writing over them file by file would leave any scan the new run did not select sitting
    there — possibly in the other frame, since a SCANNER-frame cloud and a SOURCE-frame one
    look identical — and the manifest would not mention it.

    It removes only files this extractor itself writes. ``--overwrite`` pointed at a
    directory holding anything else refuses rather than deleting it: a mistyped path must not
    be able to destroy data, and the option means "replace my last extraction", not "empty
    this directory".
    """
    import shutil

    outputs = (work_dir / "scans", work_dir / "images", work_dir / "extraction_manifest.json")
    existing = [p for p in outputs if p.exists()]
    if existing and not overwrite:
        raise ContractError(
            f"{work_dir} already holds extraction output ({[p.name for p in existing]}). "
            "Writing into it would mix two runs whose scans cannot be told apart afterwards. "
            "Use a new directory, or pass --overwrite to replace it."
        )
    for path in existing:
        if path.is_dir():
            foreign = sorted(p.name for p in path.iterdir() if not _is_extractor_output(p))
            if foreign:
                raise ContractError(
                    f"{path} holds files this extractor did not write ({foreign[:5]}). "
                    "--overwrite replaces a previous extraction; it will not delete anything "
                    "else. Point the extraction at a directory of its own."
                )
            shutil.rmtree(path)
        else:
            path.unlink()
    (work_dir / "scans").mkdir(parents=True, exist_ok=True)


_OUTPUT_NAME = re.compile(r"^(scan_\d{3}\.(ply|pose\.json)|image_\d{3}\.(jpg|png))$")


def _is_extractor_output(path: Path) -> bool:
    return path.is_file() and bool(_OUTPUT_NAME.match(path.name))


# ---------------------------------------------------------------- image extraction


def _read_blob(node: Any, rep_node: str, blob_field: str) -> bytes:
    blob = node[rep_node][blob_field]
    count = int(blob.byteCount())
    buf = np.empty(count, dtype=np.uint8)
    blob.read(buf, 0, count)
    return bytes(buf.tobytes())


def extract_image(handle: Any, asset: ImageAsset, out_dir: Path) -> tuple[Path, int, str]:
    """Write one embedded image's bytes. Returns (path, byte count, format).

    The declared blob field decides the format; the bytes must agree with it. An entry
    labelled ``jpegImage`` holding PNG bytes is a broken file, and writing ``image_000.jpg``
    that no decoder accepts is worse than saying so.
    """
    if asset.blob_field is None or asset.representation_source is None:
        raise ContractError(f"{asset.image_id} has no embedded pixel data to extract")
    node = handle.root["images2D"].get(asset.source_index)
    payload = _read_blob(node, asset.representation_source, asset.blob_field)
    magic = _BLOB_MAGIC[asset.blob_field]
    if not payload.startswith(magic):
        raise ContractError(
            f"{asset.image_id}: the entry declares {asset.blob_field} but its bytes do not "
            f"start with the {_BLOB_FORMAT[asset.blob_field]} signature. The format is not "
            "guessed from the content — inspect the file"
        )
    out = out_dir / f"{asset.image_id}{_BLOB_SUFFIX[asset.blob_field]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out, len(payload), _BLOB_FORMAT[asset.blob_field]


def _image_results(
    handle: Any,
    report: PanoMappingReport,
    work_dir: Path,
) -> tuple[list[ImageOutput], list[SkippedImage], list[str]]:
    outputs: list[ImageOutput] = []
    skipped: list[SkippedImage] = []
    issues: list[str] = []
    records = {m.image_id: m for m in report.mappings}
    for asset in report.images:
        rec = records[asset.image_id]
        if asset.source == "external_file":
            outputs.append(_external_image_output(asset, rec))
            continue
        if asset.representation not in SUPPORTED_IMAGE_REPRESENTATIONS:
            skipped.append(
                SkippedImage(
                    image_id=asset.image_id,
                    representation=asset.representation,
                    reason=(
                        f"unsupported: {asset.representation} images are not written in Phase "
                        "0B.3, which has no perspective handling (no crop, cube map, COLMAP "
                        "camera or undistortion). The entry is recorded, not reinterpreted"
                    ),
                    mapping_status=rec.status,
                )
            )
            continue
        if not asset.is_extractable:
            skipped.append(
                SkippedImage(
                    image_id=asset.image_id,
                    representation=asset.representation,
                    reason="the entry declares no jpegImage or pngImage blob; the pixels are "
                    "not in this file",
                    mapping_status=rec.status,
                )
            )
            continue
        try:
            path, size, fmt = extract_image(handle, asset, work_dir / "images")
        except ContractError as e:
            issues.append(str(e))
            skipped.append(
                SkippedImage(
                    image_id=asset.image_id,
                    representation=asset.representation,
                    reason=str(e),
                    mapping_status=rec.status,
                )
            )
            continue
        outputs.append(
            ImageOutput(
                image_id=asset.image_id,
                path=str(path),
                source="e57_embedded",
                extracted=True,
                representation=asset.representation,
                width=asset.width,
                height=asset.height,
                image_format=fmt,
                blob_field=asset.blob_field,
                bytes_written=size,
                mapped_scan_id=rec.scan_id,
                mapped_station_id=rec.station_id,
                mapping_status=rec.status,
                mapping_evidence_type=rec.evidence_type,
                sha256=sha256_file(path),
                issues=list(asset.issues),
            )
        )
    return outputs, skipped, issues


def _external_image_output(asset: ImageAsset, rec: MappingRecord) -> ImageOutput:
    """An image that is already a file is referenced, not copied.

    Copying a directory of panoramas into the staging tree would duplicate gigabytes to no
    benefit: extraction means getting data *out of* the E57, and these are already out.
    """
    path = Path(asset.path or "")
    return ImageOutput(
        image_id=asset.image_id,
        path=str(path.resolve()),
        source="external_file",
        extracted=False,
        representation=asset.representation,
        width=asset.width,
        height=asset.height,
        image_format=path.suffix.lstrip(".").lower() or None,
        mapped_scan_id=rec.scan_id,
        mapped_station_id=rec.station_id,
        mapping_status=rec.status,
        mapping_evidence_type=rec.evidence_type,
        sha256=sha256_file(path) if path.is_file() else None,
        issues=list(asset.issues),
    )


# ---------------------------------------------------------------- public entry point


def extract(
    path: str | Path,
    work_dir: str | Path,
    scan_ids: list[str] | None = None,
    voxel_m: float | None = None,
    registered: bool = True,
    with_images: bool = True,
    mapping: str | Path | None = None,
    vendor_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
    compute_hash: bool = True,
    overwrite: bool = False,
    max_scan_points: int | None = None,
) -> E57ExtractionManifest:
    """Extract scans (and, where supported, images) into a Phase 0B staging directory.

    ``registered=True`` (the default) writes points in the E57's own SOURCE frame, and every
    selected scan must declare a usable pose. ``registered=False`` writes each scan in its own
    SCANNER frame and marks the output ``unregistered``; a scan with no pose is allowed there,
    a scan with a broken one is not. The two are recorded distinctly and are never mixed in
    one run, because a scanner-frame cloud and a SOURCE-frame cloud are indistinguishable by
    inspection and catastrophic to confuse.
    """
    p = Path(path)
    work_dir = Path(work_dir)

    inv = inventory(p, compute_hash=False)
    selected, memory_notes = plan_scans(inv, scan_ids, registered, max_scan_points)
    report = build_mapping_report(
        p,
        mapping=mapping,
        vendor_manifest=vendor_manifest,
        images_dir=images_dir,
        compute_hash=False,
        inv=inv,
    )
    # Everything that can refuse has refused by now; only then is anything written.
    _prepare_work_dir(work_dir, overwrite)

    issues: list[str] = []
    notes: list[str] = list(memory_notes)
    scan_outputs: list[ScanOutput] = []
    image_outputs: list[ImageOutput] = []
    skipped: list[SkippedImage] = []

    with _nodes.open_e57(p) as handle:
        for s in selected:
            header = handle.get_header(s.scan_index)
            pose, pose_status = scan_pose(header)
            cloud, meta = read_scan_points(handle, s.scan_index, header)
            frame: ExtractionFrame = SCANNER_FRAME
            if registered:
                if pose is None or pose_status not in REGISTERABLE_POSE_STATES:
                    # plan_scans already refused this; a second gate so no future caller can
                    # reach a silent identity by skipping the preflight.
                    raise E57PoseUnusableError(
                        f"{s.scan_id} has pose status {pose_status!r} and cannot be registered"
                    )
                cloud = cloud.transformed(pose.se3(), frame=SOURCE_FRAME)
                frame = SOURCE_FRAME
            masked = len(cloud)
            if voxel_m:
                cloud = cloud.select(voxel_downsample(cloud.xyz, voxel_m))
            out_path = write_ply(cloud, work_dir / "scans" / f"{s.scan_id}.ply", xyz_dtype="f8")
            _write_pose_json(work_dir / "scans" / f"{s.scan_id}.pose.json", s, pose, frame)
            scan_outputs.append(
                ScanOutput(
                    scan_id=s.scan_id,
                    scan_index=s.scan_index,
                    path=str(out_path),
                    point_count_input=s.point_count,
                    point_count_masked=masked,
                    point_count_output=len(cloud),
                    source_frame=frame,
                    registration_status="registered" if registered else "unregistered",
                    pose_status=pose_status,
                    pose=pose,
                    voxel_m=voxel_m,
                    attributes=meta["attributes"],
                    dropped_attributes=meta["dropped_attributes"],
                    invalid_state_field=meta["invalid_state_field"],
                    invalid_points_removed=meta["invalid_points_removed"],
                    color_source_range=meta["color_source_range"],
                    color_conversion=meta["color_conversion"],
                    sha256=sha256_file(out_path),
                    issues=meta["issues"],
                    notes=[meta["coordinate_source"] + " coordinates"],
                )
            )
        if with_images:
            image_outputs, skipped, image_issues = _image_results(handle, report, work_dir)
            issues.extend(image_issues)

    inv.save(work_dir / "inventory.json")
    report.save(work_dir / "pano_mapping.json")

    issues.extend(report.issues)
    notes.extend(report.notes)
    if not with_images and report.images:
        notes.append(f"{len(report.images)} images discovered but not extracted (--no-images)")
    # Deliberately phrased without naming the later frames: a grep of any Phase 0B artifact
    # for those names must be a true positive, so not even a disclaimer may contain one.
    notes.append(
        "these outputs are in the "
        + ("SOURCE" if registered else "SCANNER")
        + " frame. Declaring a registered global frame, choosing a metric origin and building "
        "dataset/ are Phase 0C (docs/ROADMAP.md)"
    )

    sha = sha256_file(p) if compute_hash else None
    manifest = E57ExtractionManifest(
        source_e57=str(p.resolve()),
        source_sha256=sha,
        hash_skipped_reason=None if compute_hash else "requested with --no-hash",
        work_dir=str(work_dir.resolve()),
        registration="registered" if registered else "unregistered",
        output_frame=SOURCE_FRAME if registered else SCANNER_FRAME,
        scan_outputs=scan_outputs,
        image_outputs=image_outputs,
        skipped_images=skipped,
        mapping_report=report,
        issues=issues,
        notes=notes,
        provenance=ProvenanceRecord(
            git_commit=git_commit(),
            source_assets=[
                SourceAsset(path=str(p.resolve()), sha256=sha or "", size_bytes=p.stat().st_size)
            ],
            tool_versions=tool_versions(),
        ),
    )
    manifest.save(work_dir / "extraction_manifest.json")
    return manifest


def _write_pose_json(path: Path, scan: Any, pose: ScanPose | None, frame: str) -> None:
    """The scan's pose next to its points, with the direction stated by the key name."""
    path.write_text(
        json.dumps(
            {
                "scan_id": scan.scan_id,
                "scan_index": scan.scan_index,
                "guid": scan.guid,
                "name": scan.name,
                "point_frame": frame,
                "source_frame": SOURCE_FRAME,
                "pose_status": scan.pose_status,
                "registration_status": "registered" if frame == SOURCE_FRAME else "unregistered",
                "T_source_from_scanner": pose.T_source_from_scan if pose else None,
            },
            indent=2,
        )
        + "\n"
    )
