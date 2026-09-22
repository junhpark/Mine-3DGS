"""Static masks for COLMAP (``masks/<image>.png``: 0 = ignore). Nadir (tripod / operator)
for 360 crops, and rectangular exclusion boxes for fixed rigs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from minegs.ingest.common.equirect import CropView


def nadir_mask_for_crop(view: CropView, nadir_el_deg: float = -60.0) -> np.ndarray:
    """Mask (H,W) uint8, 0 where a crop pixel looks below ``nadir_el_deg`` in the scanner frame."""
    d = view.pixel_dirs()
    el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
    m = (el > nadir_el_deg).astype(np.uint8) * 255
    return m.reshape(view.spec.height, view.spec.width)


def equirect_nadir_mask(
    width: int, height: int, nadir_el_deg: float = -60.0, el_flip: bool = False
) -> np.ndarray:
    v = np.arange(height) + 0.5
    el = (0.5 - v / height) * 180.0
    if el_flip:
        el = -el
    row = (el > nadir_el_deg).astype(np.uint8) * 255
    return np.repeat(row[:, None], width, axis=1)


def box_mask(width: int, height: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    m = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 0
    return m


def write_mask(mask: np.ndarray, image_name: str, masks_dir: Path) -> Path:
    """COLMAP expects ``masks/<image_name>.png`` (the name keeps its own extension).

    ``image_name`` is relative to the image root and may contain directories — a 360 ring set
    is laid out per view — so the mask's own parent is created here. COLMAP resolves the mask
    for an image by that exact relative name, and silently applies none when it does not
    resolve, which is why the name is derived from the image rather than assembled by hand.
    """
    p = masks_dir / (image_name + ".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(p)
    return p
