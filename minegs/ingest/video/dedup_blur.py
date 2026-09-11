"""Blur scoring (variance of Laplacian) and near-duplicate removal (dHash Hamming distance).
Pure numpy so it runs in CI; fast enough for a few thousand frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)
    return img.astype(np.float64)


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian; lower = blurrier. Threshold ~60-100 for 1080p tunnel video."""
    g = gray(img)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = np.asarray(
        Image.fromarray(gray(img).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR),
        dtype=np.int16,
    )
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class FrameDecision:
    path: str
    blur: float
    hash: int
    keep: bool
    reason: str = ""


def select_frames(
    paths: list[Path],
    blur_threshold: float = 60.0,
    dedup_hamming: int = 6,
    max_frames: int | None = None,
) -> list[FrameDecision]:
    out: list[FrameDecision] = []
    last_kept: int | None = None
    kept = 0
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        b = blur_score(arr)
        h = dhash(arr)
        keep, reason = True, ""
        if b < blur_threshold:
            keep, reason = False, f"blur {b:.1f} < {blur_threshold}"
        elif last_kept is not None and hamming(h, last_kept) <= dedup_hamming:
            keep, reason = False, "duplicate of previous kept frame"
        elif max_frames is not None and kept >= max_frames:
            keep, reason = False, "max_frames reached"
        if keep:
            last_kept = h
            kept += 1
        out.append(FrameDecision(str(p), b, h, keep, reason))
    return out
