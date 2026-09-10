"""PDAL pipelines for large cartesian scans (§6.1): voxel downsample + chainage/XY tiling.
Pipelines are built as JSON (pure python, testable) and executed via the pdal python
bindings or the ``pdal`` CLI — whichever is present."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from minegs.core.errors import MissingDependencyError


def downsample_pipeline(
    src: str | Path, dst: str | Path, voxel_m: float = 0.02, reader: str | None = None
) -> dict[str, Any]:
    stages: list[Any] = [{"type": reader or _reader_for(src), "filename": str(src)}]
    if voxel_m and voxel_m > 0:
        stages.append({"type": "filters.voxelcenternearestneighbor", "cell": voxel_m})
    stages.append(
        {
            "type": "writers.ply",
            "filename": str(dst),
            "storage_mode": "little endian",
            "precision": 4,
        }
    )
    return {"pipeline": stages}


def tile_pipeline(
    src: str | Path,
    out_dir: str | Path,
    length_m: float = 80.0,
    buffer_m: float = 15.0,
    voxel_m: float | None = 0.02,
) -> dict[str, Any]:
    """Square XY tiles (PDAL splitter). Chainage-based chunking is done in python on top."""
    stages: list[Any] = [{"type": _reader_for(src), "filename": str(src)}]
    if voxel_m:
        stages.append({"type": "filters.voxelcenternearestneighbor", "cell": voxel_m})
    stages.append({"type": "filters.splitter", "length": length_m, "buffer": buffer_m})
    stages.append(
        {
            "type": "writers.ply",
            "filename": str(Path(out_dir) / "tile_#.ply"),
            "storage_mode": "little endian",
        }
    )
    return {"pipeline": stages}


def _reader_for(src: str | Path) -> str:
    ext = Path(src).suffix.lower()
    return {
        ".e57": "readers.e57",
        ".ply": "readers.ply",
        ".las": "readers.las",
        ".laz": "readers.las",
    }.get(ext, "readers.ply")


def run_pipeline(pipeline: dict[str, Any], workdir: str | Path | None = None) -> int:
    """Execute with python-pdal if importable, else the pdal CLI. Returns point count."""
    try:
        import pdal

        p = pdal.Pipeline(json.dumps(pipeline))
        return int(p.execute())
    except ImportError:
        pass
    exe = shutil.which("pdal")
    if not exe:
        raise MissingDependencyError("pdal", "e57", "PDAL tiling")
    wd = Path(workdir or ".")
    pf = wd / "pipeline.json"
    pf.write_text(json.dumps(pipeline, indent=2))
    subprocess.run([exe, "pipeline", str(pf)], check=True)
    return -1
