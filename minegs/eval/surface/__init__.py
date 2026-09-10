"""GS -> surface (§1.7). Gaussian centres are *not* a surface; extract depth/TSDF/mesh first."""

from minegs.eval.surface.depth import backproject_depth, depth_to_points
from minegs.eval.surface.mesh import mesh_volume, sample_mesh_surface

__all__ = ["backproject_depth", "depth_to_points", "mesh_volume", "sample_mesh_surface"]
