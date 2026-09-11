"""TSDF fusion of rendered depth maps (open3d), then marching cubes -> mesh (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from minegs.core.errors import MissingDependencyError


def fuse_tsdf(
    depth_dir: str | Path, sparse_dir: str | Path, voxel_m: float = 0.03, trunc_m: float = 0.12
) -> Path:
    try:
        import open3d  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError("open3d", "eval", "TSDF fusion") from e
    from minegs.core.errors import NotYetImplementedError

    raise NotYetImplementedError("TSDF fusion", "1")
