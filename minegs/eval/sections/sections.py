"""Cross-sections along the centerline (§11): A(s) from a point cloud.

At each station s a slab of thickness ``thickness_m`` orthogonal to the tangent is taken,
points are expressed in the section frame (y right, z up), binned by polar angle around the
centerline, and the wall radius per bin is the median of that bin. The polygon through the
bin radii gives the area (shoelace). Bins with no points mark the section invalid when more
than ``max_missing_bins`` are empty — missing sections are *reported*, not interpolated.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.centerline import Centerline


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chainage_m: float
    area_m2: float | None
    valid: bool
    n_points: int
    empty_bins: int
    radii_m: list[float | None]  # per angle bin, None = empty
    center: list[float]  # section origin in the cloud frame
    tangent: list[float]


class SectionSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: str
    interval_m: float
    thickness_m: float
    angle_bins: int
    start_chainage_m: float
    end_chainage_m: float
    sections: list[Section] = Field(default_factory=list)

    def chainages(self) -> np.ndarray:
        return np.array([s.chainage_m for s in self.sections])

    def areas(self, fill: float = np.nan) -> np.ndarray:
        return np.array(
            [s.area_m2 if s.valid and s.area_m2 is not None else fill for s in self.sections]
        )

    def valid_count(self) -> int:
        return sum(s.valid for s in self.sections)


def polygon_area(radii: np.ndarray, angles: np.ndarray) -> float:
    y = radii * np.cos(angles)
    z = radii * np.sin(angles)
    return float(0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1))))


def extract_sections(
    points: np.ndarray,
    centerline: Centerline,
    interval_m: float = 1.0,
    thickness_m: float = 0.1,
    angle_bins: int = 180,
    start_m: float | None = None,
    end_m: float | None = None,
    max_missing_bins: int | None = None,
    min_points: int = 20,
    frame: str = "TLS_GLOBAL",
) -> SectionSeries:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    s_all, _ = centerline.project(pts)
    stations = centerline.stations(interval_m, start_m, end_m)
    max_missing = angle_bins // 10 if max_missing_bins is None else max_missing_bins
    edges = np.linspace(0, 2 * np.pi, angle_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(s_all)
    s_sorted = s_all[order]
    series = SectionSeries(
        frame=frame,
        interval_m=interval_m,
        thickness_m=thickness_m,
        angle_bins=angle_bins,
        start_chainage_m=float(stations[0]) if len(stations) else 0.0,
        end_chainage_m=float(stations[-1]) if len(stations) else 0.0,
    )
    for s in stations:
        lo = np.searchsorted(s_sorted, s - thickness_m / 2)
        hi = np.searchsorted(s_sorted, s + thickness_m / 2)
        idx = order[lo:hi]
        T = centerline.frame_at(float(s))
        local = T.inverse().apply(pts[idx]) if len(idx) else np.zeros((0, 3))
        radii: list[float | None] = [None] * angle_bins
        area = None
        valid = False
        empty = angle_bins
        if len(idx) >= min_points:
            ang = np.mod(np.arctan2(local[:, 2], local[:, 1]), 2 * np.pi)
            r = np.hypot(local[:, 1], local[:, 2])
            b = np.minimum((ang / (2 * np.pi) * angle_bins).astype(int), angle_bins - 1)
            sums = np.zeros(angle_bins)
            for k in range(angle_bins):
                m = b == k
                if m.any():
                    sums[k] = np.median(r[m])
                    radii[k] = float(sums[k])
            empty = int(sum(v is None for v in radii))
            if empty <= max_missing:
                filled = np.array([v if v is not None else np.nan for v in radii])
                if empty:  # interpolate the few missing bins circularly
                    good = ~np.isnan(filled)
                    filled = np.interp(centers, centers[good], filled[good], period=2 * np.pi)
                area = polygon_area(filled, centers)
                valid = True
        series.sections.append(
            Section(
                chainage_m=float(s),
                area_m2=area,
                valid=valid,
                n_points=len(idx),
                empty_bins=empty,
                radii_m=radii,
                center=T.t.tolist(),
                tangent=T.R[:, 0].tolist(),
            )
        )
    return series
