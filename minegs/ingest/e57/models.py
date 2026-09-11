"""E57 inventory contract (Phase 0B.1).

pye57 / libE57 objects never leave ``inventory.py``. Everything downstream reads the models
below, so a vendor quirk or a pye57 API change is absorbed in one adapter.

Two rules shape these models:

* **Inspection first, interpretation second.** Nothing here assumes one scan is one station,
  that a pose exists, that colour exists, or that the file's coordinates mean anything in
  particular. Optional things are ``None`` and the reason is recorded, never guessed.
* **Frames are named, not assumed.** An E57's own coordinates are ``SOURCE``. Declaring them
  TLS_GLOBAL is a decision made later, during dataset materialisation (ARCHITECTURE.md §3).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.frames import SE3
from minegs.core.provenance import ProvenanceRecord
from minegs.ingest.e57.exceptions import E57PoseUnusableError

#: Frame label for an E57's own coordinates. NOT TLS_GLOBAL — that mapping is decided later.
SOURCE_FRAME = "SOURCE"

#: How a station candidate came to exist. Phase 0B.1 can only infer one-per-scan; Phase 0B.2
#: confirms (or corrects) it with panorama evidence.
MappingStatus = Literal["inferred_from_scan", "confirmed", "manual"]
#: What the file gave us for a scan's pose.
#: ``absent``     - no pose node at all (an unregistered scan; not an error)
#: ``unreadable`` - a pose node exists but could not be parsed (a problem, NOT "absent")
#: ``invalid``    - parsed, but not a rigid transform
#: ``identity``   - valid, but the file declares no displacement
#: ``valid``      - usable
PoseStatus = Literal["absent", "unreadable", "invalid", "identity", "valid"]
StationOrigin = Literal["e57_scan", "derived", "manual"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: A rotation deviating by less than this is treated as a *rounded* value (vendors commonly
#: store 4-6 decimal places), recorded as a warning and orthonormalised for the transform.
POSE_WARN_TOL = 1e-9
#: Beyond this the rotation is not a rotation: the pose is invalid, not merely imprecise.
POSE_INVALID_TOL = 1e-4


class PoseValidation(_Strict):
    """Why a pose is or is not usable.

    Two tiers, because "stored with four decimal places" and "not a rotation matrix" are
    different facts and collapsing them makes the report useless on real files:

    * ``issues`` — the pose is not usable as a rigid transform (``valid=False``).
    * ``warnings`` — usable, but something was worth saying, e.g. the quaternion was slightly
      off unit length and was orthonormalised. Never silent: the measured deviation is here.
    """

    valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    quaternion_norm: float | None = None
    rotation_determinant: float | None = None
    orthonormality_error: float | None = None
    normalised: bool = False


class ScanPose(_Strict):
    """A scan's rigid placement inside the E57's own coordinate system.

    ``T_source_from_scan`` maps a point expressed in the scanner's own frame into the file's
    SOURCE frame. The name states the direction; ``transform`` / ``matrix`` / ``pose_matrix``
    would not.
    """

    T_source_from_scan: list[list[float]]
    translation_m: list[float]
    rotation_quaternion_wxyz: list[float]
    source_frame: str = SOURCE_FRAME
    unit: Literal["m"] = "m"
    is_identity: bool = False
    validation: PoseValidation

    def se3(self) -> SE3:
        """The pose as an ``SE3``.

        Fail-closed: a pose whose ``validation`` failed is never handed out as a working
        transform. Without this gate a scan declaring only a translation would come back as a
        perfectly ordinary identity-rotation SE(3), which is exactly the silent fallback the
        inventory exists to prevent.
        """
        if not self.validation.valid:
            raise E57PoseUnusableError("; ".join(self.validation.issues) or "validation failed")
        return SE3.from_matrix(self.T_source_from_scan)


class ScanBounds(_Strict):
    """Axis-aligned bounds as declared in the scan header, in the scan's own frame.

    Read from header metadata only. ``None`` when the header declares none — minegs does not
    stream millions of points to compute them during an inventory.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    source_field: str = "cartesianBounds"
    valid: bool = True
    issues: list[str] = Field(default_factory=list)

    def extent(self) -> tuple[float, float, float]:
        return (self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z)


class E57ScanInventory(_Strict):
    """What one Data3D scan declares about itself."""

    scan_index: int
    scan_id: str
    name: str | None = None
    guid: str | None = None
    point_count: int | None = None

    # minegs semantic flags — what a caller can actually rely on
    has_cartesian_xyz: bool = False
    has_spherical: bool = False
    has_rgb: bool = False
    has_intensity: bool = False
    has_row_column: bool = False

    #: The file declares a pose node. Says nothing about whether it could be read or is valid.
    pose_declared: bool = False
    #: Collapses declared/parsed/valid into the one word a reader actually needs.
    pose_status: PoseStatus = "absent"
    #: The parsed pose. ``None`` when absent OR unreadable — check ``pose_status`` to tell
    #: those apart; a pose node that failed to parse must never look like no pose at all.
    pose: ScanPose | None = None
    bounds: ScanBounds | None = None

    # library-level detail, kept for diagnosis and provenance, not for logic
    raw_point_fields: list[str] = Field(default_factory=list)
    raw_scan_fields: list[str] = Field(default_factory=list)
    vendor_metadata: dict[str, Any] | None = None

    #: Problems: this scan cannot be used as declared. Never fatal to the inventory itself.
    issues: list[str] = Field(default_factory=list)
    #: Observations worth telling the user, but not problems (identity pose, rounded rotation).
    notes: list[str] = Field(default_factory=list)

    @property
    def is_usable_for_points(self) -> bool:
        """Whether point positions could be reconstructed at all (either coordinate system)."""
        return self.has_cartesian_xyz or self.has_spherical

    @property
    def pose_is_broken(self) -> bool:
        """The file declares a pose but it cannot be used. Absent is not broken."""
        return self.pose_status in ("unreadable", "invalid")

    @property
    def pose_is_usable(self) -> bool:
        return self.pose is not None and self.pose.validation.valid


class StationCandidate(_Strict):
    """A *candidate* capture station.

    Phase 0B.1 can only say "this scan looks like a station". It deliberately does not claim
    a confirmed station/panorama relationship: ``mapping_status`` stays ``inferred_from_scan``
    until Phase 0B.2 brings panorama evidence.
    """

    station_id: str
    scan_ids: list[str]
    origin: StationOrigin = "e57_scan"
    mapping_status: MappingStatus = "inferred_from_scan"
    note: str | None = None


class ImageSummary(_Strict):
    """Detection only. Phase 0B.1 never decodes or extracts an image (Phase 0B.2 does).

    ``image_count`` is ``None`` when the structure exists but could not be enumerated. An
    unknown count must not be reported as zero: "present but unreadable" and "absent" are
    different facts about the file, and flattening them is the interpretation this phase
    refuses to make.
    """

    has_images2d: bool = False
    image_count: int | None = 0
    enumeration_status: Literal["ok", "absent", "error"] = "absent"
    detection_note: str | None = None


class E57FileInfo(_Strict):
    path: str
    file_name: str
    size_bytes: int
    sha256: str | None = None
    hash_skipped_reason: str | None = None
    e57_library_version: str | None = None
    format_name: str | None = None
    guid: str | None = None
    coordinate_metadata: str | None = None


class E57Inventory(VersionedModel):
    """``minegs ingest e57 inventory`` result — a report, not a dataset.

    Nothing here has been promoted into the dataset contract: no LOCAL_METRIC origin, no
    TLS_GLOBAL declaration, no capture groups. Those are later phases (docs/ROADMAP.md).
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    file: E57FileInfo
    scan_count: int
    scans: list[E57ScanInventory] = Field(default_factory=list)
    station_candidates: list[StationCandidate] = Field(default_factory=list)
    images: ImageSummary = Field(default_factory=ImageSummary)
    #: File-level problems (e.g. "3 of 8 scans declare an invalid pose").
    issues: list[str] = Field(default_factory=list)
    #: File-level observations (e.g. "no images2D structure").
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    def scan_by_id(self, scan_id: str) -> E57ScanInventory:
        for s in self.scans:
            if s.scan_id == scan_id:
                return s
        raise KeyError(scan_id)

    def scans_with_usable_pose(self) -> list[E57ScanInventory]:
        return [s for s in self.scans if s.pose_is_usable]

    def has_any_issue(self) -> bool:
        return bool(self.issues) or any(s.issues for s in self.scans)

    def usable_scan_count(self) -> int:
        """Scans that declare a coordinate triple and whose pose, if declared, is usable.

        A scan with no pose at all is usable (it is simply unregistered). A scan whose pose
        node exists but could not be read or is not a rotation is not.
        """
        return sum(1 for s in self.scans if s.is_usable_for_points and not s.pose_is_broken)


# ---------------------------------------------------------------- deterministic identifiers


def scan_id_for(index: int) -> str:
    """``0 -> "scan_000"``, from the scan's index inside its source file.

    Independent of vendor metadata (name, GUID), so a file that omits those still gets stable
    references; it is *not* independent of the order scans appear in the file. Paired with the
    file's SHA-256 in provenance, that is enough to identify a scan unambiguously.
    """
    if index < 0:
        raise ValueError("scan index must be >= 0")
    return f"scan_{index:03d}"


def station_id_for(index: int) -> str:
    """``0 -> "S000"``, matching ``scan_id_for`` one-to-one for the inferred default."""
    if index < 0:
        raise ValueError("station index must be >= 0")
    return f"S{index:03d}"


# ---------------------------------------------------------------- pose construction


def validate_rotation(R: np.ndarray, quat_wxyz: np.ndarray | None = None) -> PoseValidation:
    """Check a rotation the way §10 asks: finite, orthonormal, det ≈ +1, unit quaternion.

    Deviations are graded against ``POSE_INVALID_TOL`` (broken) and ``POSE_WARN_TOL``
    (rounded); see ``PoseValidation``.
    """
    issues: list[str] = []
    warnings: list[str] = []
    q_norm: float | None = None
    det: float | None = None
    orth_err: float | None = None

    if quat_wxyz is not None:
        q = np.asarray(quat_wxyz, dtype=np.float64)
        if q.size != 4:
            issues.append(f"quaternion has {q.size} components, expected 4")
        elif not np.all(np.isfinite(q)):
            issues.append("quaternion has non-finite components")
        else:
            q_norm = float(np.linalg.norm(q))
            dev = abs(q_norm - 1.0)
            if q_norm == 0.0:
                issues.append("quaternion has zero norm")
            elif dev > POSE_INVALID_TOL:
                issues.append(f"quaternion is not unit length (norm {q_norm:.9f})")
            elif dev > POSE_WARN_TOL:
                warnings.append(
                    f"quaternion norm is {q_norm:.9f}, off unit by {dev:.2e} "
                    "(looks rounded; normalised for the transform)"
                )

    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        issues.append(f"rotation is not 3x3 (shape {R.shape})")
        return PoseValidation(valid=False, issues=issues, warnings=warnings, quaternion_norm=q_norm)
    if not np.all(np.isfinite(R)):
        issues.append("rotation has non-finite components")
        return PoseValidation(valid=False, issues=issues, warnings=warnings, quaternion_norm=q_norm)

    orth_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    if orth_err > POSE_INVALID_TOL:
        issues.append(f"rotation is not orthonormal (max |RtR - I| = {orth_err:.3e})")
    elif orth_err > POSE_WARN_TOL:
        warnings.append(f"rotation is orthonormal only to {orth_err:.2e}; orthonormalised")
    if abs(det - 1.0) > POSE_INVALID_TOL:
        issues.append(
            f"rotation determinant is {det:.9f}, expected +1"
            + (" (a determinant near -1 means the rotation is mirrored)" if det < 0 else "")
        )

    return PoseValidation(
        valid=not issues,
        issues=issues,
        warnings=warnings,
        quaternion_norm=q_norm,
        rotation_determinant=det,
        orthonormality_error=orth_err,
    )


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to R (SVD), so a rounded quaternion still yields a valid SE3."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1
        out = U @ Vt
    return out


def make_scan_pose(
    R: np.ndarray,
    t: np.ndarray,
    quat_wxyz: np.ndarray | None = None,
    extra_issues: list[str] | None = None,
) -> ScanPose:
    """Build a ``ScanPose`` from a rotation and translation, validating both.

    An invalid pose is still returned, with ``validation.valid = False`` and the reasons. It
    is never silently replaced by identity — a caller that needs a usable transform must check
    ``validation.valid`` and decide, which is the whole point of reporting it.

    A rotation that is merely *rounded* is orthonormalised so the stored transform is a real
    SE(3), and the fact plus the measured deviation are recorded in ``validation``.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    validation = validate_rotation(R, quat_wxyz)
    # Structural problems found by the caller (e.g. the file declares a pose with no rotation)
    # make the pose invalid even though the numbers we fell back on look fine: an undeclared
    # rotation is unknown, not identity.
    issues = list(validation.issues) + list(extra_issues or [])

    if t.shape != (3,):
        issues.append(f"translation is not 3-vector (shape {t.shape})")
        t = np.zeros(3)
    elif not np.all(np.isfinite(t)):
        issues.append("translation has non-finite components")
        t = np.where(np.isfinite(t), t, 0.0)

    R_out = R
    normalised = False
    usable = R.shape == (3, 3) and np.all(np.isfinite(R))
    # Only a pose that is otherwise sound gets orthonormalised (a rounded rotation made
    # usable). A pose with real problems keeps the matrix the file actually declared: turning
    # 3*I into I here would manufacture exactly the plausible-looking identity §10 forbids.
    if (
        usable
        and not issues
        and validation.orthonormality_error
        and validation.orthonormality_error > POSE_WARN_TOL
    ):
        R_out = orthonormalise(R)
        normalised = True

    if issues != validation.issues or normalised:
        validation = validation.model_copy(
            update={"issues": issues, "valid": not issues, "normalised": normalised}
        )

    M = np.eye(4)
    if usable:
        M[:3, :3] = R_out
    M[:3, 3] = t

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1) if quat_wxyz is not None else None
    is_identity = bool(
        np.allclose(M[:3, :3], np.eye(3), atol=1e-9) and np.allclose(t, 0.0, atol=1e-12)
    )
    return ScanPose(
        T_source_from_scan=M.tolist(),
        translation_m=t.tolist(),
        rotation_quaternion_wxyz=(q.tolist() if q is not None and q.size == 4 else []),
        is_identity=is_identity,
        validation=validation,
    )


def make_bounds(
    values: dict[str, float], source_field: str = "cartesianBounds"
) -> ScanBounds | None:
    """Build bounds from header numbers, reporting rather than trusting them.

    Writers do get this wrong (pye57's own writer emits min > max), so a header that declares
    impossible bounds is recorded as ``valid=False`` instead of being propagated as fact.
    """
    keys = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
    if not all(k in values for k in keys):
        return None
    nums = {k: float(values[k]) for k in keys}
    issues: list[str] = []
    if not all(math.isfinite(v) for v in nums.values()):
        issues.append("bounds contain non-finite values")
    else:
        for axis in ("x", "y", "z"):
            if nums[f"min_{axis}"] > nums[f"max_{axis}"]:
                issues.append(
                    f"{axis}: min ({nums[f'min_{axis}']:g}) exceeds max ({nums[f'max_{axis}']:g})"
                )
    return ScanBounds(**nums, source_field=source_field, valid=not issues, issues=issues)
