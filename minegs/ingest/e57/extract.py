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
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.pointcloud import PointCloud, voxel_downsample, write_ply
from minegs.core.provenance import (
    ProvenanceRecord,
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
    with_source_hash,
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
    #: Why ``sha256`` is absent, so "not hashed" never reads as "could not be hashed".
    hash_skipped_reason: str | None = None
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


#: What this extractor writes at the root of a staging directory.
_ROOT_OUTPUTS = ("inventory.json", "pano_mapping.json", "extraction_manifest.json")
_OUTPUT_DIRS = {"scans": re.compile(r"^scan_\d{3}\.(ply|pose\.json)$")}
_OUTPUT_DIRS["images"] = re.compile(r"^image_\d{3}\.(jpg|png)$")


def foreign_entries(work_dir: Path) -> list[str]:
    """Everything in ``work_dir`` that this extractor would not have written.

    Used to decide whether a directory is ours to replace. Anything unrecognised — a stray
    note, a hand-edited JSON, a subdirectory of someone's own — means the directory is not
    ours, and neither ``--overwrite`` nor the publish step may touch it.
    """
    out: list[str] = []
    for entry in sorted(work_dir.iterdir(), key=lambda e: e.name):
        pattern = _OUTPUT_DIRS.get(entry.name)
        if pattern is not None and entry.is_dir():
            out += [
                f"{entry.name}/{c.name}"
                for c in sorted(entry.iterdir(), key=lambda c: c.name)
                if not (c.is_file() and pattern.match(c.name))
            ]
        elif not (entry.name in _ROOT_OUTPUTS and entry.is_file()):
            out.append(entry.name + ("/" if entry.is_dir() else ""))
    return out


def _recover_interrupted_publish(work_dir: Path) -> str | None:
    """Put back a tree that a crash left moved aside, and say so.

    :func:`_publish` moves the old tree to ``.<name>.minegs-previous``, renames the new one
    into place, then removes the backup. A crash between the first two steps leaves no
    ``work_dir`` at all and a backup holding the only copy of the previous run. Restoring it
    makes that window self-healing; treating the backup as stale and discarding it would turn
    a crash into silent data loss, which is the whole failure mode this design exists to
    remove.
    """
    previous = work_dir.parent / f".{work_dir.name}.minegs-previous"
    if work_dir.exists() or not previous.exists() or not previous.is_dir():
        return None
    previous.rename(work_dir)
    return (
        f"restored {work_dir} from {previous.name}: a previous publish was interrupted after "
        "the old tree had been moved aside but before the new one was in place"
    )


def _prepare_target(work_dir: Path, overwrite: bool) -> str | None:
    """Recover an interrupted publish, then decide whether we may publish into ``work_dir``.

    Called twice: once at the very top of a run, so a mistyped target costs nothing, and again
    from :func:`staging_dir` at the moment it matters. Idempotent — the second call finds the
    recovery already done.
    """
    note = _recover_interrupted_publish(work_dir)
    _check_target(work_dir, overwrite)
    return note


def _check_target(work_dir: Path, overwrite: bool) -> None:
    """Whether ``work_dir`` is ours to replace. Pure: it inspects and refuses, nothing else.

    A directory holding anything this extractor did not write is refused first and regardless
    of ``overwrite`` — that refusal protects data, and the occupied-directory one only
    protects tidiness.
    """
    if not work_dir.name:
        raise ContractError(
            f"{work_dir} has no directory name to stage beside. Give the extraction a named "
            "directory of its own."
        )
    if work_dir.is_symlink():
        # Publishing renames the target aside, which would replace the *link* and leave what
        # it points at untouched — the opposite of what --overwrite says it does.
        raise ContractError(
            f"{work_dir} is a symlink. Extraction publishes by renaming the target into "
            "place, which would replace the link rather than what it points at. Pass the "
            "real directory."
        )
    if work_dir.exists() and not work_dir.is_dir():
        raise ContractError(
            f"{work_dir} exists and is not a directory. Extraction writes a staging tree, so "
            "it needs a directory of its own."
        )
    if not work_dir.exists() or not any(work_dir.iterdir()):
        return
    foreign = foreign_entries(work_dir)
    if foreign:
        raise ContractError(
            f"{work_dir} holds files this extractor did not write ({foreign[:5]}). Extraction "
            "replaces a whole staging tree, and it will not delete anything else. Point it at "
            "a directory of its own."
        )
    if not overwrite:
        raise ContractError(
            f"{work_dir} already holds extraction output. Writing into it would mix two runs "
            "whose scans cannot be told apart afterwards. Use a new directory, or pass "
            "--overwrite to replace it."
        )


def _discard(path: Path, what: str) -> None:
    """Remove a directory this extractor owns, refusing if anything else is inside it."""
    if not path.exists():
        return
    if not path.is_dir():
        raise ContractError(f"{path} is {what}, but it is not a directory. Remove it and re-run.")
    foreign = foreign_entries(path)
    if foreign:
        raise ContractError(
            f"{path} is {what} and holds files this extractor did not write ({foreign[:5]}). "
            "Inspect it and remove it yourself; this tool will not delete it."
        )
    shutil.rmtree(path)


@contextmanager
def staging_dir(work_dir: Path, overwrite: bool) -> Iterator[Path]:
    """Write a run into a sibling temporary tree and publish it only once it is complete.

    Extraction writes point clouds one scan at a time and the manifest last. Writing those
    straight into ``work_dir`` means a failure on scan 12 of 40 leaves twelve scans, no
    manifest, and a directory that looks like a finished run to anything that only checks for
    PLYs — and with ``--overwrite`` it means the previous *good* run was already deleted to
    make room for the one that just failed.

    So the run lands in ``.<name>.minegs-partial`` next to the target (same directory, hence
    the same filesystem, hence an atomic rename) and is moved into place at the end. On the
    way there the old tree is moved aside rather than deleted, and put back if the final
    rename fails. A failure anywhere leaves ``work_dir`` exactly as it was.
    """
    _prepare_target(work_dir, overwrite)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = work_dir.parent / f".{work_dir.name}.minegs-partial"
    _discard(tmp, "left over from an interrupted extraction")
    (tmp / "scans").mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        _publish(tmp, work_dir)
    except ContractError:
        raise
    except OSError as e:
        # The run itself succeeded, so the completed tree is kept and named rather than
        # deleted; only the move failed. ContractError keeps this on the exit-code contract.
        raise ContractError(
            f"the extraction completed but could not be published into {work_dir} ({e}). "
            f"Nothing there was changed; the finished run is at {tmp}."
        ) from e


def _publish(tmp: Path, work_dir: Path) -> None:
    """Move a complete staging tree into place, restoring the previous one if that fails."""
    if not work_dir.exists():
        tmp.rename(work_dir)
        return
    previous = work_dir.parent / f".{work_dir.name}.minegs-previous"
    _discard(previous, "left over from an interrupted publish")
    work_dir.rename(previous)
    try:
        tmp.rename(work_dir)
    except OSError as e:
        previous.rename(work_dir)
        raise ContractError(
            f"the extraction completed but could not be moved into {work_dir} ({e}). The "
            f"previous contents are back in place and the new run is at {tmp}."
        ) from e
    # The run is published; failing it now over a leftover backup would be a lie about what
    # happened. The next run's target check finds and removes it.
    shutil.rmtree(previous, ignore_errors=True)


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
    staged: Path,
    work_dir: Path,
) -> tuple[list[ImageOutput], list[SkippedImage], list[str]]:
    """Write each supported image into ``staged``, recording its published ``work_dir`` path."""
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
            path, size, fmt = extract_image(handle, asset, staged / "images")
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
                path=str(work_dir / "images" / path.name),
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
        # The digest discovery already computed for this file — hashing it a second time
        # would double the I/O and could disagree if the file changed underneath us.
        sha256=asset.sha256,
        hash_skipped_reason=asset.hash_skipped_reason,
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

    # Cheapest refusal first: a mistyped or occupied target should not cost a 50 GB read. The
    # staging context checks again at the moment it matters, so a race here is not a hole.
    recovery_note = _prepare_target(work_dir, overwrite)

    inv = inventory(p, compute_hash=False)
    selected, memory_notes = plan_scans(inv, scan_ids, registered, max_scan_points)
    # One digest for the whole staging tree. Hashing is O(file size), so it happens exactly
    # once and only after the preflight — a refusal should not first stream 50 GB. All three
    # artifacts written below then agree about which bytes they describe, which is what makes
    # an index-derived id like scan_000 mean anything.
    sha = sha256_file(p) if compute_hash else None
    inv = with_source_hash(inv, sha)
    report = build_mapping_report(
        p,
        mapping=mapping,
        vendor_manifest=vendor_manifest,
        images_dir=images_dir,
        compute_hash=False,
        source_sha256=sha,
        inv=inv,
    )
    issues: list[str] = []
    notes: list[str] = ([recovery_note] if recovery_note else []) + list(memory_notes)
    scan_outputs: list[ScanOutput] = []
    image_outputs: list[ImageOutput] = []
    skipped: list[SkippedImage] = []

    # Everything that can refuse from metadata alone has refused by now. What follows can
    # still fail on a scan's payload, so it lands in a temporary tree and is published whole.
    with staging_dir(work_dir, overwrite) as staged, _nodes.open_e57(p) as handle:
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
            out_path = write_ply(cloud, staged / "scans" / f"{s.scan_id}.ply", xyz_dtype="f8")
            _write_pose_json(staged / "scans" / f"{s.scan_id}.pose.json", s, pose, frame)
            scan_outputs.append(
                ScanOutput(
                    scan_id=s.scan_id,
                    scan_index=s.scan_index,
                    # The path this file will have once the tree is published, not the
                    # temporary one it is being written to.
                    path=str(work_dir / "scans" / f"{s.scan_id}.ply"),
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
            image_outputs, skipped, image_issues = _image_results(handle, report, staged, work_dir)
            issues.extend(image_issues)

        inv.save(staged / "inventory.json")
        report.save(staged / "pano_mapping.json")

        issues.extend(report.issues)
        notes.extend(report.notes)
        if not with_images and report.images:
            notes.append(f"{len(report.images)} images discovered but not extracted (--no-images)")
        # Deliberately phrased without naming the later frames: a grep of any Phase 0B
        # artifact for those names must be a true positive, so not even a disclaimer may
        # contain one.
        notes.append(
            "these outputs are in the "
            + ("SOURCE" if registered else "SCANNER")
            + " frame. Declaring a registered global frame, choosing a metric origin and "
            "building dataset/ are Phase 0C (docs/ROADMAP.md)"
        )

        manifest = E57ExtractionManifest(
            source_e57=str(p.resolve()),
            source_sha256=sha,
            hash_skipped_reason=None if sha is not None else "requested with --no-hash",
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
                # The mapping report already enumerated every input that shaped this run —
                # the E57, each mapping file, the external image set. Rebuilding the list
                # here would let the two artifacts drift about what was read.
                source_assets=list(report.provenance.source_assets),
                tool_versions=tool_versions(),
            ),
        )
        # Written last, inside the staging tree: a directory holding a manifest is a complete
        # run, and one is only ever published whole.
        manifest.save(staged / "extraction_manifest.json")

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
