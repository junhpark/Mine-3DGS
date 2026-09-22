"""Coordinate frames (§3): SOURCE -> TLS_GLOBAL -> LOCAL_METRIC -> BACKEND_INTERNAL.

* ``SE3``  — rigid (R, t). TLS_GLOBAL <-> LOCAL_METRIC is *always* SE3 (1 unit = 1 m).
* ``Sim3`` — similarity (s, R, t). Only appears in registration of SfM output (§7).
* Naming: ``T_a_from_b`` maps points expressed in frame b into frame a.
* ``float32_precision_m`` explains why UTM-scale coordinates never go to the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from minegs.core.errors import FrameError

ArrayLike = Any


class Frame(str, Enum):
    SOURCE = "SOURCE"
    #: An independent image/360 SfM reconstruction's own coordinates (Phase 3 AD-1). Not
    #: ``SOURCE``: a scanner's file is already in metres, while this is arbitrary in scale as
    #: well as in origin, and the door out of ``SOURCE`` is a *declaration* that returns an
    #: SE(3) and structurally cannot carry scale. Keeping the two apart is what stops an
    #: unscaled reconstruction from being called metric by writing one line of config. The only
    #: exit is a measured Sim(3) recorded in a registration artifact.
    SFM_INTERNAL = "SFM_INTERNAL"
    TLS_GLOBAL = "TLS_GLOBAL"
    LOCAL_METRIC = "LOCAL_METRIC"
    BACKEND_INTERNAL = "BACKEND_INTERNAL"


#: Frames whose unit is the metre and whose origin is tied to the survey (§3). Everything that
#: measures distance lives in one of these.
METRIC_FRAMES = frozenset({Frame.TLS_GLOBAL.value, Frame.LOCAL_METRIC.value})


def require_metric_frame(frame: str, what: str) -> None:
    """Refuse geometry that is not in a metric frame, naming ``SFM_INTERNAL`` when it is one.

    Most consumers already compare against the exact frame they need, so this adds nothing for
    them. It exists for the one case worth a sentence of its own: an arbitrary-scale SfM
    reconstruction arriving where metres are expected. Distances computed from it would be in
    no unit at all, and the failure is silent unless someone says so.
    """
    if frame == Frame.SFM_INTERNAL.value:
        raise FrameError(
            f"{what} is in {frame}, an independent SfM reconstruction's own arbitrary-scale "
            "coordinates. It has no metric meaning until a measured Sim(3) registration puts "
            "it in TLS_GLOBAL (Phase 3 §7); until then every length taken from it is unitless."
        )
    if frame not in METRIC_FRAMES:
        raise FrameError(f"{what} is in {frame}, not a metric frame ({sorted(METRIC_FRAMES)})")


# |x| * 2^-23 : spacing of adjacent float32 values at magnitude |x|
FLOAT32_EPS = 2.0**-23
FLOAT32_SAFE_MAGNITUDE_M = 5_000.0  # ~0.6 mm resolution; beyond this we warn/refuse


def float32_precision_m(magnitude_m: float) -> float:
    """Resolution (m) of float32 at the given coordinate magnitude (m)."""
    return abs(float(magnitude_m)) * FLOAT32_EPS


def check_float32_safe(points_or_t: ArrayLike, what: str = "coordinates") -> float:
    """Return max |coord|; raise FrameError if float32 would lose sub-cm precision."""
    arr = np.asarray(points_or_t, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mag = float(np.max(np.abs(arr)))
    if mag > FLOAT32_SAFE_MAGNITUDE_M:
        raise FrameError(
            f"{what}: max |coord| = {mag:.1f} m -> float32 resolution "
            f"{float32_precision_m(mag) * 1000:.1f} mm. Express in LOCAL_METRIC (§3) first."
        )
    return mag


def _as_rotation(R: ArrayLike, tol: float = 1e-6) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise FrameError(f"rotation must be 3x3, got {R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=tol) or not math.isclose(
        float(np.linalg.det(R)), 1.0, abs_tol=tol
    ):
        raise FrameError("rotation is not orthonormal with det +1")
    return R


def quat_to_rotmat(q: ArrayLike) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation (COLMAP convention)."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise FrameError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(R: ArrayLike) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass(frozen=True)
class SE3:
    """Rigid transform: ``p_a = R @ p_b + t``."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _as_rotation(self.R))
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        object.__setattr__(self, "t", t)

    @classmethod
    def identity(cls) -> SE3:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, M: ArrayLike) -> SE3:
        M = np.asarray(M, dtype=np.float64)
        if M.shape != (4, 4):
            raise FrameError(f"expected 4x4 matrix, got {M.shape}")
        if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
            raise FrameError("last row of SE3 matrix must be [0 0 0 1]")
        return cls(M[:3, :3], M[:3, 3])

    @classmethod
    def from_translation(cls, t: ArrayLike) -> SE3:
        return cls(np.eye(3), np.asarray(t, dtype=np.float64))

    @classmethod
    def from_quat_t(cls, q_wxyz: ArrayLike, t: ArrayLike) -> SE3:
        return cls(quat_to_rotmat(q_wxyz), t)

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def to_list(self) -> list[list[float]]:
        return self.matrix().tolist()

    def quat(self) -> np.ndarray:
        return rotmat_to_quat(self.R)

    def inverse(self) -> SE3:
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t)

    def __matmul__(self, other: SE3) -> SE3:
        if not isinstance(other, SE3):
            return NotImplemented
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, points: ArrayLike) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p = p.reshape(-1, 3)
        out = p @ self.R.T + self.t
        return out[0] if single else out

    def apply_dirs(self, dirs: ArrayLike) -> np.ndarray:
        d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        return d @ self.R.T

    def is_identity(self, tol: float = 1e-9) -> bool:
        return bool(np.allclose(self.R, np.eye(3), atol=tol) and np.allclose(self.t, 0, atol=tol))

    def allclose(self, other: SE3, atol: float = 1e-6) -> bool:
        return bool(
            np.allclose(self.R, other.R, atol=atol) and np.allclose(self.t, other.t, atol=atol)
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"SE3(t={np.round(self.t, 4).tolist()}, q={np.round(self.quat(), 5).tolist()})"


@dataclass(frozen=True)
class Sim3:
    """Similarity transform: ``p_a = s * R @ p_b + t`` (§7 registration only)."""

    s: float
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        if not (self.s > 0 and math.isfinite(self.s)):
            raise FrameError(f"Sim3 scale must be positive finite, got {self.s}")
        object.__setattr__(self, "s", float(self.s))
        object.__setattr__(self, "R", _as_rotation(self.R))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).reshape(3))

    @classmethod
    def identity(cls) -> Sim3:
        return cls(1.0, np.eye(3), np.zeros(3))

    @classmethod
    def from_se3(cls, T: SE3, s: float = 1.0) -> Sim3:
        return cls(s, T.R, T.t)

    @classmethod
    def from_matrix(cls, M: ArrayLike) -> Sim3:
        M = np.asarray(M, dtype=np.float64)
        A = M[:3, :3]
        s = float(np.cbrt(np.linalg.det(A)))
        return cls(s, A / s, M[:3, 3])

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M

    def to_list(self) -> list[list[float]]:
        return self.matrix().tolist()

    def inverse(self) -> Sim3:
        Rt = self.R.T
        return Sim3(1.0 / self.s, Rt, -(Rt @ self.t) / self.s)

    def __matmul__(self, other: Sim3 | SE3) -> Sim3:
        if isinstance(other, SE3):
            other = Sim3.from_se3(other)
        if not isinstance(other, Sim3):
            return NotImplemented
        return Sim3(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def apply(self, points: ArrayLike) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p = p.reshape(-1, 3)
        out = self.s * (p @ self.R.T) + self.t
        return out[0] if single else out

    def is_identity(self, tol: float = 1e-9) -> bool:
        return bool(
            abs(self.s - 1) < tol
            and np.allclose(self.R, np.eye(3), atol=tol)
            and np.allclose(self.t, 0, atol=tol)
        )

    def se3(self, scale_tol: float = 1e-6) -> SE3:
        """Drop the scale. Raises unless |s-1| <= tol — scale must be resolved explicitly."""
        if abs(self.s - 1.0) > scale_tol:
            raise FrameError(f"Sim3 has scale {self.s:.6f}; resolve scale before treating as SE3")
        return SE3(self.R, self.t)

    def se3_part(self) -> SE3:
        """(R, t) part with scale intentionally discarded (for reporting only)."""
        return SE3(self.R, self.t)


def se3_from_any(value: ArrayLike | SE3 | None) -> SE3:
    if value is None:
        return SE3.identity()
    if isinstance(value, SE3):
        return value
    return SE3.from_matrix(value)
