"""GS -> surface (§1.7). Gaussian centres are *not* a surface; extract depth/TSDF/mesh first."""

from minegs.eval.surface.depth import backproject_depth, build_depth_surface, depth_to_points
from minegs.eval.surface.mesh import mesh_volume, sample_mesh_surface
from minegs.eval.surface.models import SurfaceRecord, check_surface, find_surface, load_surface

__all__ = [
    "SurfaceRecord",
    "backproject_depth",
    "build_depth_surface",
    "check_surface",
    "depth_to_points",
    "find_surface",
    "load_surface",
    "mesh_volume",
    "sample_mesh_surface",
]
