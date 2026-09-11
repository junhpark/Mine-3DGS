"""Equirectangular -> pinhole *ring crops* (§6.1, §6.2).

A ring crop is a set of virtual pinhole cameras sharing the panorama's optical centre,
at yaw steps around the vertical axis (optionally several pitches). Since the intrinsics
are synthetic they are known exactly (``fix_intrinsics: true``) and all crops of one frame
form a COLMAP rig (``rig.py``).

Camera convention (COLMAP/OpenCV): x right, y down, z forward. ``R_scanner_from_cam``
maps camera rays into the scanner/pano frame (z up).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.frames import rot_y, rot_z
from minegs.ingest.common.geometry import PanoConvention, dirs_to_equirect_uv


class RingCropSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_yaw: int = Field(default=8, ge=1)
    fov_deg: float = Field(default=90.0, gt=0, lt=180)
    pitches_deg: list[float] = Field(default_factory=lambda: [0.0])
    width: int = Field(default=1600, ge=16)
    height: int = Field(default=1600, ge=16)
    yaw_offset_deg: float = 0.0

    def K(self) -> np.ndarray:
        f = 0.5 * self.width / np.tan(np.radians(self.fov_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])

    def crops(self) -> list[CropView]:
        out = []
        for pi, pitch in enumerate(self.pitches_deg):
            for yi in range(self.n_yaw):
                yaw = self.yaw_offset_deg + 360.0 * yi / self.n_yaw
                out.append(CropView(f"p{pi}y{yi:02d}", yaw, pitch, self))
        return out


# base: camera looking along +x of scanner with x_cam=-y_scanner? Build explicitly:
# cam axes in scanner frame at yaw=0, pitch=0: forward(z_cam)=+x, right(x_cam)=-y, down(y_cam)=-z
_R_SCANNER_FROM_CAM0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass
class CropView:
    name: str
    yaw_deg: float
    pitch_deg: float
    spec: RingCropSpec = field(repr=False)

    @property
    def R_scanner_from_cam(self) -> np.ndarray:
        # yaw about scanner z, then pitch about the camera's right axis (scanner -y at yaw 0)
        return rot_z(self.yaw_deg) @ rot_y(-self.pitch_deg) @ _R_SCANNER_FROM_CAM0

    def pixel_dirs(self) -> np.ndarray:
        """(H*W,3) unit ray directions in the scanner frame for every crop pixel."""
        K = self.spec.K()
        w, h = self.spec.width, self.spec.height
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        d = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d @ self.R_scanner_from_cam.T


def crop_equirect(
    pano: np.ndarray, view: CropView, conv: PanoConvention | None = None, order: int = 1
) -> np.ndarray:
    """Resample an equirect image (H,W[,C]) into the crop's pinhole image via bilinear lookup."""
    from scipy.ndimage import map_coordinates

    H, W = pano.shape[:2]
    uv = dirs_to_equirect_uv(view.pixel_dirs(), W, H, conv)
    # pixel centres: continuous coord u in [0,W) maps to sample index u-0.5
    cols = np.mod(uv[:, 0] - 0.5, W)
    rows = np.clip(uv[:, 1] - 0.5, 0, H - 1)
    coords = np.vstack([rows, cols])
    h, w = view.spec.height, view.spec.width
    if pano.ndim == 2:
        out = map_coordinates(pano.astype(np.float32), coords, order=order, mode="wrap")
        return out.reshape(h, w).astype(pano.dtype)
    chans = [
        map_coordinates(pano[..., c].astype(np.float32), coords, order=order, mode="wrap").reshape(
            h, w
        )
        for c in range(pano.shape[2])
    ]
    out = np.stack(chans, axis=-1)
    if pano.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255)
    return out.astype(pano.dtype)
