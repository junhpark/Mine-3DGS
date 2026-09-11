"""Depth-map based surface samples. Rendering depth needs the backend (gsplat rasteriser);
this module only turns depth maps + poses into LOCAL_METRIC points."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minegs.core.errors import NotYetImplementedError
from minegs.core.frames import SE3
from minegs.core.pointcloud import PointCloud


def backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    world_from_cam: SE3,
    stride: int = 1,
    max_depth: float | None = None,
) -> np.ndarray:
    """Depth (H,W, metres along z) -> world points (N,3)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    m = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        m &= z <= max_depth
    x = (u[m] + 0.5 - K[0, 2]) / K[0, 0] * z[m]
    y = (v[m] + 0.5 - K[1, 2]) / K[1, 1] * z[m]
    return world_from_cam.apply(np.column_stack([x, y, z[m]]))


def depth_to_points(
    depth_dir: str | Path,
    cameras: dict,
    images: dict,
    stride: int = 2,
    max_depth: float | None = None,
) -> PointCloud:
    """Fuse ``<depth_dir>/<image_name>.npy`` maps (rendered by the backend) into one cloud."""
    depth_dir = Path(depth_dir)
    pts = []
    for im in images.values():
        f = depth_dir / (Path(im.name).stem + ".npy")
        if not f.exists():
            continue
        pts.append(
            backproject_depth(
                np.load(f), cameras[im.camera_id].K(), im.world_from_cam, stride, max_depth
            )
        )
    if not pts:
        raise FileNotFoundError(f"no depth maps in {depth_dir}")
    return PointCloud(np.concatenate(pts), frame="LOCAL_METRIC")


def render_depths(run_dir: str | Path, dataset_dir: str | Path, out_dir: str | Path) -> Path:
    """Render depth per training/test view with the run's backend. GPU only (Phase 0D/3)."""
    raise NotYetImplementedError("depth rendering from a trained run", "1")
