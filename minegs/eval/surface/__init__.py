"""GS -> surface (§1.7). Gaussian centres are *not* a surface; extract depth/TSDF/mesh first."""

from minegs.eval.surface.depth import backproject_depth, build_depth_surface, depth_to_points
from minegs.eval.surface.mesh import mesh_volume, sample_mesh_surface
from minegs.eval.surface.models import (
    DepthManifest,
    SurfaceRecord,
    check_surface,
    find_depth_manifest,
    find_surface,
    load_surface,
    verify_depth_manifest,
)
from minegs.eval.surface.render import DepthRenderer, get_depth_renderer, render_depths

__all__ = [
    "DepthManifest",
    "DepthRenderer",
    "SurfaceRecord",
    "backproject_depth",
    "build_depth_surface",
    "check_surface",
    "depth_to_points",
    "find_depth_manifest",
    "find_surface",
    "get_depth_renderer",
    "load_surface",
    "mesh_volume",
    "render_depths",
    "sample_mesh_surface",
    "verify_depth_manifest",
]
