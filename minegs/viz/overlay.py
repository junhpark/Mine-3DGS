"""2D overlay for convention validation (golden gate, §6.1/§13 0C): reproject scanner-frame
points onto the panorama and draw them; a wrong ``PanoConvention`` is obvious at a glance,
and the best of the 8 discrete candidates can be picked automatically by edge agreement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

from minegs.ingest.common.geometry import (
    PanoConvention,
    convention_candidates,
    reproject_points_to_pano,
)


def colormap_turbo_like(x: np.ndarray) -> np.ndarray:
    """Cheap perceptual ramp (blue -> green -> yellow -> red) for values in [0,1]."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render_overlay(
    pano: np.ndarray,
    xyz_scanner: np.ndarray,
    conv: PanoConvention,
    max_range: float = 40.0,
    alpha: float = 0.8,
    subsample: int = 1,
) -> np.ndarray:
    H, W = pano.shape[:2]
    uv, r = reproject_points_to_pano(xyz_scanner[::subsample], W, H, conv, max_range)
    out = pano.copy()
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    u = np.clip(uv[:, 0].astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].astype(int), 0, H - 1)
    col = colormap_turbo_like(r / max_range)
    out[v, u] = (alpha * col + (1 - alpha) * out[v, u]).astype(np.uint8)
    return out


def range_image(
    xyz_scanner: np.ndarray, width: int, height: int, conv: PanoConvention, max_range: float = 40.0
) -> np.ndarray:
    """Nearest-point range panorama (float32, NaN where empty)."""
    uv, r = reproject_points_to_pano(xyz_scanner, width, height, conv, max_range)
    img = np.full((height, width), np.inf, dtype=np.float32)
    u = np.clip(uv[:, 0].astype(int), 0, width - 1)
    v = np.clip(uv[:, 1].astype(int), 0, height - 1)
    np.minimum.at(img, (v, u), r.astype(np.float32))
    img[~np.isfinite(img)] = np.nan
    return img


def edge_agreement(
    pano: np.ndarray, xyz_scanner: np.ndarray, conv: PanoConvention, scale: int = 4
) -> float:
    """Correlation between image edges and range-image edges at reduced resolution."""
    H, W = pano.shape[0] // scale, pano.shape[1] // scale
    g = np.asarray(Image.fromarray(pano).convert("L").resize((W, H)), dtype=np.float64)
    rng_img = range_image(xyz_scanner, W, H, conv)
    rng_img = np.where(np.isnan(rng_img), np.nanmedian(rng_img), rng_img)
    e1 = np.hypot(sobel(gaussian_filter(g, 1), 0), sobel(gaussian_filter(g, 1), 1))
    e2 = np.hypot(sobel(gaussian_filter(rng_img, 1), 0), sobel(gaussian_filter(rng_img, 1), 1))
    e1, e2 = e1 - e1.mean(), e2 - e2.mean()
    denom = np.sqrt((e1**2).sum() * (e2**2).sum())
    return float((e1 * e2).sum() / denom) if denom > 0 else 0.0


def calibrate_convention(
    pano: np.ndarray, xyz_scanner: np.ndarray, refine_offset: bool = True
) -> tuple[PanoConvention, dict[str, float]]:
    """Pick the best of the 8 discrete conventions, optionally refine az_offset by 1° steps."""
    scores: dict[str, float] = {}
    best, best_s = None, -np.inf
    for c in convention_candidates():
        s = edge_agreement(pano, xyz_scanner, c)
        scores[f"sign={c.az_sign} flip={c.el_flip} off={c.az_offset_deg}"] = s
        if s > best_s:
            best, best_s = c, s
    assert best is not None
    if refine_offset:
        for off in np.arange(-15, 16, 1.0):
            c = PanoConvention(
                best.az_sign, best.el_flip, best.az_offset_deg + off, best.source, best.vendor
            )
            s = edge_agreement(pano, xyz_scanner, c)
            if s > best_s:
                best, best_s = c, s
    scores["best"] = best_s
    return best, scores


def save_overlay(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path
