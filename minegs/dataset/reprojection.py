"""Pinhole reprojection helpers shared by calibration and the Golden Gate (§12, §25).

Everything here is deliberately elementary — project, z-buffer, compare — so the overlay
shows the *geometry* and nothing hides an error: an axis flip, a 90° turn, a mirrored image,
a displaced centre or a wrong focal length each moves the projected points somewhere a
human can see is wrong. "The points fall inside the image bounds" is not a pass (§25).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, sobel

from minegs.core.frames import SE3
from minegs.ingest.common import colmap_io


@dataclass
class Projection:
    uv: np.ndarray  # (N,2) continuous pixel coords, all points
    depth: np.ndarray  # (N,) camera-frame z
    inside: np.ndarray  # (N,) bool: in front and within bounds

    @property
    def n_inside(self) -> int:
        return int(self.inside.sum())


def project_points(
    K: np.ndarray, cam_from_world: SE3, xyz_world: np.ndarray, width: int, height: int
) -> Projection:
    uv, z = colmap_io.project(K, cam_from_world, xyz_world)
    inside = colmap_io.visible_mask(uv, z, width, height, near=1e-3)
    return Projection(uv, z, inside)


def depth_image(proj: Projection, width: int, height: int) -> np.ndarray:
    """Nearest depth per pixel (float32, NaN where empty)."""
    img = np.full((height, width), np.inf, dtype=np.float32)
    u = proj.uv[proj.inside, 0].astype(int)
    v = proj.uv[proj.inside, 1].astype(int)
    np.minimum.at(img, (v, u), proj.depth[proj.inside].astype(np.float32))
    img[~np.isfinite(img)] = np.nan
    return img


def rgb_residual(image: np.ndarray, proj: Projection, point_rgb: np.ndarray) -> tuple[float, int]:
    """Mean |image − point| over channels at the projected pixels (0–255); lower is better.

    No occlusion handling: the points compared are the station's own scan, seen from the
    station's own camera, so nearly everything the scanner saw the camera saw too.
    """
    if proj.n_inside == 0:
        return float("inf"), 0
    u = proj.uv[proj.inside, 0].astype(int)
    v = proj.uv[proj.inside, 1].astype(int)
    px = image[v, u].astype(np.float32)
    pc = point_rgb[proj.inside].astype(np.float32)
    return float(np.mean(np.abs(px - pc))), proj.n_inside


def edge_agreement(image: np.ndarray, depth: np.ndarray, scale: int = 4) -> float:
    """Correlation between image edges and depth-image edges at reduced resolution (−1..1)."""
    H, W = max(1, image.shape[0] // scale), max(1, image.shape[1] // scale)
    g = np.asarray(PILImage.fromarray(image).convert("L").resize((W, H)), dtype=np.float64)
    d = np.asarray(
        PILImage.fromarray(
            np.nan_to_num(
                depth, nan=float(np.nanmedian(depth)) if np.isfinite(depth).any() else 0.0
            )
        ).resize((W, H)),
        dtype=np.float64,
    )
    e1 = np.hypot(sobel(gaussian_filter(g, 1), 0), sobel(gaussian_filter(g, 1), 1))
    e2 = np.hypot(sobel(gaussian_filter(d, 1), 0), sobel(gaussian_filter(d, 1), 1))
    e1, e2 = e1 - e1.mean(), e2 - e2.mean()
    denom = float(np.sqrt((e1**2).sum() * (e2**2).sum()))
    return float((e1 * e2).sum() / denom) if denom > 0 else 0.0


def colormap(x: np.ndarray) -> np.ndarray:
    """Blue → green → yellow → red for values in [0, 1]."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render_overlay(
    image: np.ndarray,
    proj: Projection,
    colour: np.ndarray,
    alpha: float = 0.85,
    radius: int = 1,
) -> np.ndarray:
    """Paint projected points onto the image (RGB uint8). ``colour`` is (N,3) uint8 per point."""
    out = image.copy()
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    H, W = out.shape[:2]
    u = proj.uv[proj.inside, 0].astype(int)
    v = proj.uv[proj.inside, 1].astype(int)
    c = colour[proj.inside]
    # far points first so near ones paint over them, like a z-buffer
    order = np.argsort(-proj.depth[proj.inside])
    u, v, c = u[order], v[order], c[order]
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            out[vv, uu] = (alpha * c + (1 - alpha) * out[vv, uu]).astype(np.uint8)
    return out


def depth_colours(proj: Projection, max_depth: float | None = None) -> np.ndarray:
    d = proj.depth
    lim = max_depth or (float(np.percentile(d[proj.inside], 98)) if proj.n_inside else 1.0)
    return colormap(np.clip(d, 0, lim) / max(lim, 1e-9))


def save_image(path, image: np.ndarray) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(image).save(p)
