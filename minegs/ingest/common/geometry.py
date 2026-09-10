"""Spherical <-> cartesian and panorama (equirectangular) projection with an explicit
``PanoConvention`` (az_sign, el_flip, az_offset). Vendors disagree on where azimuth 0 is and
which way it turns; §6.1 makes the convention a calibrated, manifest-recorded quantity and
the reprojection check (points -> panorama pixels) is the *golden gate* of Phase 0C.

Scanner frame: right-handed, z up. Azimuth measured in the xy-plane from +x, elevation
from the xy-plane towards +z. A pixel column u spans azimuth over [0, 2π), row v spans
elevation from +π/2 (top) to -π/2 (bottom).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanoConvention:
    az_sign: int = 1  # +1: azimuth increases counter-clockwise (seen from +z) with u
    el_flip: bool = False  # True: row 0 is nadir instead of zenith
    az_offset_deg: float = 0.0  # azimuth at pixel column 0
    source: str = "E57Embedded"
    vendor: str | None = None

    def to_manifest(self) -> dict:
        return {
            "az_sign": self.az_sign,
            "el_flip": self.el_flip,
            "az_offset": self.az_offset_deg,
            "source": self.source,
            "vendor": self.vendor,
        }

    @classmethod
    def from_manifest(cls, d: dict) -> PanoConvention:
        return cls(
            int(d.get("az_sign", 1)),
            bool(d.get("el_flip", False)),
            float(d.get("az_offset", 0.0)),
            str(d.get("source", "E57Embedded")),
            d.get("vendor"),
        )


def cart_to_spherical(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xyz -> (range, azimuth [0,2π), elevation [-π/2, π/2])."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(p, axis=1)
    az = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi)
    el = np.arcsin(np.clip(p[:, 2] / np.where(r == 0, 1, r), -1, 1))
    return r, az, el


def spherical_to_cart(r: np.ndarray, az: np.ndarray, el: np.ndarray) -> np.ndarray:
    r, az, el = (np.asarray(v, dtype=np.float64) for v in (r, az, el))
    c = np.cos(el)
    return np.column_stack([r * c * np.cos(az), r * c * np.sin(az), r * np.sin(el)])


def dirs_to_equirect_uv(
    dirs: np.ndarray, width: int, height: int, conv: PanoConvention | None = None
) -> np.ndarray:
    """Unit (or any) directions in scanner frame -> continuous pixel coords (u, v)."""
    conv = conv or PanoConvention()
    _, az, el = cart_to_spherical(dirs)
    az = np.mod(conv.az_sign * (az - np.radians(conv.az_offset_deg)), 2 * np.pi)
    u = az / (2 * np.pi) * width
    v = (0.5 - el / np.pi) * height
    if conv.el_flip:
        v = height - v
    return np.column_stack([u, v])


def equirect_uv_to_dirs(
    uv: np.ndarray, width: int, height: int, conv: PanoConvention | None = None
) -> np.ndarray:
    conv = conv or PanoConvention()
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]
    if conv.el_flip:
        v = height - v
    az = conv.az_sign * (u / width * 2 * np.pi) + np.radians(conv.az_offset_deg)
    el = (0.5 - v / height) * np.pi
    return spherical_to_cart(np.ones_like(az), az, el)


def reproject_points_to_pano(
    xyz_scanner: np.ndarray,
    width: int,
    height: int,
    conv: PanoConvention | None = None,
    max_range: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Golden-gate check helper: scanner-frame points -> (uv, range). Used by viz.overlay."""
    r, _, _ = cart_to_spherical(xyz_scanner)
    uv = dirs_to_equirect_uv(xyz_scanner, width, height, conv)
    if max_range is not None:
        keep = r <= max_range
        return uv[keep], r[keep]
    return uv, r


def convention_candidates() -> list[PanoConvention]:
    """The 8 discrete conventions to try during calibration (offset is continuous, refined after)."""
    return [PanoConvention(s, f, o) for s in (1, -1) for f in (False, True) for o in (0.0, 180.0)]
