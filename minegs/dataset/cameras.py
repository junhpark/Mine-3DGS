"""E57 pinhole images → COLMAP cameras and poses (§9–§11, §15).

Intrinsics come from what the E57 declares and nothing else. The field semantics are those
of the E57 ``PinholeRepresentation`` as documented in libE57Format (``E57SimpleData.h``)::

    focalLength      camera focal length, metres
    pixelWidth       width of a pixel, metres          -> fx = focalLength / pixelWidth
    pixelHeight      height of a pixel, metres         -> fy = focalLength / pixelHeight
    principalPointX  X of the principal point, pixels  -> cx (intersection of the camera
    principalPointY  Y of the principal point, pixels  -> cy  frame's z axis with the image)
    imageWidth / imageHeight   pixels

The specification says the optical axis is the camera frame's z axis. It does not say, in
any material we could reach, whether the camera looks along +z or −z nor which way image y
runs. That is exactly the *axis convention* this module keeps explicit:
``R_e57cam_from_cam`` maps COLMAP/OpenCV camera axes (x right, y down, z forward) into the
E57 image frame. It is one of the 24 axis-aligned proper rotations, it comes from a
calibration artifact or an explicit configuration, never from a file name, a face index or
a default, and it is a genuine rotation (RᵀR = I, det +1).

The chain a COLMAP pose is built from (§15)::

    T_local_from_cam = T_local_from_tls @ T_tls_from_source @ T_source_from_e57cam @ R_e57cam_from_cam

``T_source_from_e57cam`` is the image's own E57 pose (``Image2D.pose``: "the coordinate frame
of the camera in the file-level coordinate system"), read from the asset's vendor metadata.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, quat_to_rotmat
from minegs.ingest.common import colmap_io
from minegs.ingest.e57.images import ImageAsset
from minegs.ingest.e57.models import POSE_INVALID_TOL, orthonormalise, validate_rotation

REQUIRED_PINHOLE_FIELDS = (
    "focalLength",
    "pixelWidth",
    "pixelHeight",
    "principalPointX",
    "principalPointY",
)

ConventionSource = Literal["calibrated", "explicit"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------- convention


def axis_aligned_rotations() -> list[np.ndarray]:
    """All 24 signed axis permutations with det +1 (proper rotations), fixed order."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                out.append(M)
    assert len(out) == 24
    return out


def convention_label(R_e57cam_from_cam: np.ndarray) -> str:
    """``cam(+X,-Y,-Z)``: the COLMAP camera's x, y, z axes expressed as E57 image-frame axes.

    Columns of ``R_e57cam_from_cam`` are the camera axes in the E57 frame, so the label reads
    "camera x is E57 +X, camera y is E57 −Y, camera z is E57 −Z" — the same spelling the
    exploratory calibration used, which is what a reader will compare against.
    """
    R = np.asarray(R_e57cam_from_cam, dtype=np.float64)
    names = "XYZ"
    parts = []
    for c in range(3):
        col = R[:, c]
        r = int(np.argmax(np.abs(col)))
        parts.append(("-" if col[r] < 0 else "+") + names[r])
    return f"cam({','.join(parts)})"


class CameraConvention(_Strict):
    """E57 image frame ← COLMAP camera frame, plus where it came from."""

    R_e57cam_from_cam: list[list[float]]
    label: str = ""
    source: ConventionSource
    #: For ``calibrated``: the artifact it was read from; for ``explicit``: the config note.
    origin: str | None = None

    @field_validator("R_e57cam_from_cam")
    @classmethod
    def _rotation(cls, v: list[list[float]]) -> list[list[float]]:
        R = np.asarray(v, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R_e57cam_from_cam must be 3x3, got {R.shape}")
        if not np.all(np.isfinite(R)):
            raise ValueError("R_e57cam_from_cam has non-finite entries")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
            raise ValueError("R_e57cam_from_cam is not orthonormal")
        if abs(np.linalg.det(R) - 1.0) > 1e-9:
            raise ValueError(
                f"R_e57cam_from_cam has det {np.linalg.det(R):.6f}; a camera convention is a "
                "proper rotation, never a reflection"
            )
        return v

    def matrix(self) -> np.ndarray:
        return np.asarray(self.R_e57cam_from_cam, dtype=np.float64)

    def with_label(self) -> CameraConvention:
        return self.model_copy(update={"label": convention_label(self.matrix())})


# ---------------------------------------------------------------------------- intrinsics


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def key(self) -> tuple[float, ...]:
        return (self.fx, self.fy, self.cx, self.cy, float(self.width), float(self.height))


def pinhole_intrinsics(asset: ImageAsset) -> PinholeIntrinsics:
    """Intrinsics from the declared E57 fields only. Anything missing is a refusal (§9)."""
    if asset.representation != "pinhole":
        raise ContractError(
            f"{asset.image_id}: representation is {asset.representation!r}, not pinhole"
        )
    meta = asset.vendor_metadata or {}
    missing = [k for k in REQUIRED_PINHOLE_FIELDS if meta.get(k) is None]
    if missing or asset.width is None or asset.height is None:
        lack = missing + [
            k for k, v in (("imageWidth", asset.width), ("imageHeight", asset.height)) if v is None
        ]
        raise ContractError(
            f"{asset.image_id}: the E57 pinholeRepresentation declares no {lack}. Intrinsics "
            "are read, never assumed — no FOV, focal length or principal point is guessed."
        )
    f = float(meta["focalLength"])
    pw, ph = float(meta["pixelWidth"]), float(meta["pixelHeight"])
    cx, cy = float(meta["principalPointX"]), float(meta["principalPointY"])
    w, h = int(asset.width), int(asset.height)
    vals = (f, pw, ph, cx, cy)
    if not all(np.isfinite(vals)):
        raise ContractError(
            f"{asset.image_id}: non-finite intrinsic field in "
            f"{dict(zip(REQUIRED_PINHOLE_FIELDS, vals, strict=True))}"
        )
    if f <= 0 or pw <= 0 or ph <= 0:
        raise ContractError(
            f"{asset.image_id}: focalLength/pixelWidth/pixelHeight must be positive "
            f"(got {f}, {pw}, {ph})"
        )
    if w <= 0 or h <= 0:
        raise ContractError(f"{asset.image_id}: image dimensions must be positive, got {w}x{h}")
    fx, fy = f / pw, f / ph
    if not (0.0 <= cx <= w and 0.0 <= cy <= h):
        raise ContractError(
            f"{asset.image_id}: principal point ({cx}, {cy}) lies outside the {w}x{h} image"
        )
    return PinholeIntrinsics(fx, fy, cx, cy, w, h)


# ---------------------------------------------------------------------------- image pose


def image_pose_source_from_e57cam(asset: ImageAsset) -> SE3:
    """``Image2D.pose`` as an SE3 in the SOURCE frame, validated like a scan pose (§10).

    A pose that is merely rounded is orthonormalised (and would have been reported by the
    inventory); a pose that is not a rotation, or absent, is refused. Nothing about the image
    (name, index) substitutes for it.
    """
    meta = asset.vendor_metadata or {}
    q = meta.get("pose_rotation_wxyz")
    t = meta.get("pose_translation")
    if q is None or t is None:
        raise ContractError(
            f"{asset.image_id}: the E57 entry declares no image pose (rotation + translation). "
            "A pinhole image without a pose cannot be placed; its file name is not evidence."
        )
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if (
        q.shape != (4,)
        or t.shape != (3,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(t))
    ):
        raise ContractError(f"{asset.image_id}: image pose is malformed or non-finite")
    if abs(float(np.linalg.norm(q)) - 1.0) > POSE_INVALID_TOL:
        raise ContractError(
            f"{asset.image_id}: image pose quaternion has norm {np.linalg.norm(q):.6f}, not 1"
        )
    R = quat_to_rotmat(q)
    v = validate_rotation(R, q)
    if not v.valid:
        raise ContractError(f"{asset.image_id}: image pose rejected: {'; '.join(v.issues)}")
    if v.orthonormality_error and v.orthonormality_error > 1e-9:
        R = orthonormalise(R)
    return SE3(R, t)


# ---------------------------------------------------------------------------- COLMAP


def colmap_pose_local_from_cam(
    T_local_from_source: SE3, T_source_from_e57cam: SE3, convention: CameraConvention
) -> SE3:
    """§15, with every direction in the name."""
    return T_local_from_source @ T_source_from_e57cam @ SE3(convention.matrix(), np.zeros(3))


class CameraTable:
    """Deduplicates identical intrinsics into one COLMAP camera id, in first-seen order."""

    def __init__(self) -> None:
        self.cameras: dict[int, colmap_io.Camera] = {}
        self._ids: dict[tuple[float, ...], int] = {}

    def id_for(self, intr: PinholeIntrinsics) -> int:
        key = intr.key()
        if key not in self._ids:
            cid = len(self.cameras) + 1
            self._ids[key] = cid
            self.cameras[cid] = colmap_io.Camera.pinhole(cid, intr.K(), intr.width, intr.height)
        return self._ids[key]


# ---------------------------------------------------------------------------- calibration artifact


class CandidateScore(_Strict):
    label: str
    rgb_residual: float | None = None
    n_points: int = 0


class StationCalibration(_Strict):
    station_id: str
    scan_id: str
    image_ids: list[str]
    best_label: str
    runner_up_label: str | None = None
    best_score: float
    runner_up_score: float | None = None
    margin: float | None = None
    candidates: list[CandidateScore]


class CameraCalibration(VersionedModel):
    """``camera_convention.json`` — the calibrated axis convention with its evidence (§12)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    status: Literal["selected", "ambiguous", "inconsistent", "insufficient"]
    convention: CameraConvention | None = None
    scoring: Literal["rgb_residual"]
    candidates: list[CandidateScore]
    best_score: float
    runner_up_score: float | None = None
    margin: float | None = None
    min_margin: float
    stations: list[StationCalibration]
    sampled_station_ids: list[str]
    sampled_image_ids: list[str]
    n_candidates: int = 24
    notes: list[str] = Field(default_factory=list)
    source_sha256: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    tool_version: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    def require_selected(self, path: str, source_sha256: str | None = None) -> CameraConvention:
        """The convention a build may use.

        Ambiguity is refused, not resolved by first match; and the artifact must have been
        measured on the *same source bytes* the build consumes — a convention calibrated on
        another survey is evidence about that survey.
        """
        if source_sha256 is not None and self.source_sha256 != source_sha256:
            raise ContractError(
                f"{path}: this calibration was measured on source {self.source_sha256[:12]}…, "
                f"but the staging tree comes from {source_sha256[:12]}…. Re-run "
                "`minegs dataset calibrate-camera` on this tree, or declare "
                "R_e57cam_from_cam explicitly."
            )
        if self.status != "selected" or self.convention is None:
            worst = "; ".join(self.notes) or "no note recorded"
            raise ContractError(
                f"{path}: camera calibration status is {self.status!r} ({worst}). A dataset is "
                "not built on a convention the evidence could not single out — sample more "
                "stations, or declare R_e57cam_from_cam explicitly and take responsibility."
            )
        return self.convention.model_copy(update={"source": "calibrated", "origin": path})
