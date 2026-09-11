"""Comparison views (§12): per-point distance heatmap colours and section polylines."""

from __future__ import annotations

import numpy as np

from minegs.eval.geometry.metrics import nn_distances
from minegs.eval.sections.sections import Section
from minegs.viz.overlay import colormap_turbo_like


def distance_heatmap(
    pred: np.ndarray, ref: np.ndarray, max_dist_m: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances, rgb uint8) for ``pred`` coloured by distance to ``ref``."""
    d = nn_distances(pred, ref)
    return d, colormap_turbo_like(d / max_dist_m)


def section_polyline(section: Section, T_line_from_section: np.ndarray | None = None) -> np.ndarray:
    """Closed 3D polyline of a section (in the cloud frame) from its per-bin radii."""
    n = len(section.radii_m)
    ang = (np.arange(n) + 0.5) / n * 2 * np.pi
    r = np.array([v if v is not None else np.nan for v in section.radii_m])
    good = ~np.isnan(r)
    if good.sum() < 3:
        return np.zeros((0, 3))
    r = np.interp(ang, ang[good], r[good], period=2 * np.pi)
    t = np.asarray(section.tangent)
    up = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    y = np.cross(up, t)
    y /= np.linalg.norm(y)
    z = np.cross(t, y)
    c = np.asarray(section.center)
    pts = c + np.outer(r * np.cos(ang), y) + np.outer(r * np.sin(ang), z)
    return np.vstack([pts, pts[:1]])
