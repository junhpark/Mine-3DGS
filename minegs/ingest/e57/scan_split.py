"""Per-station scan extraction (pye57): scanner-frame points + the scanner pose.
Spherical-only scans are converted to cartesian in the scanner frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minegs.core.frames import SE3
from minegs.core.pointcloud import PointCloud, voxel_downsample, write_ply
from minegs.ingest.common.geometry import spherical_to_cart
from minegs.ingest.e57.inventory import _pye57


def read_scan(path: str | Path, index: int, voxel_m: float | None = None) -> tuple[PointCloud, SE3]:
    """Return (cloud in SCANNER frame, T_tls_from_scanner)."""
    pye57 = _pye57()
    e57 = pye57.E57(str(path))
    try:
        data = e57.read_scan_raw(index)
        h = e57.get_header(index)
        if "cartesianX" in data:
            xyz = np.column_stack(
                [data["cartesianX"], data["cartesianY"], data["cartesianZ"]]
            ).astype(np.float64)
        else:
            xyz = spherical_to_cart(
                data["sphericalRange"], data["sphericalAzimuth"], data["sphericalElevation"]
            )
        if "cartesianInvalidState" in data:
            xyz = xyz[np.asarray(data["cartesianInvalidState"]) == 0]
        rgb = None
        if "colorRed" in data:
            rgb = np.column_stack([data["colorRed"], data["colorGreen"], data["colorBlue"]])
            rgb = rgb[: len(xyz)]
        pose = SE3(np.asarray(h.rotation_matrix), np.asarray(h.translation))
    finally:
        e57.close()
    pc = PointCloud(xyz, rgb, frame="SOURCE")
    if voxel_m:
        pc = pc.select(voxel_downsample(pc.xyz, voxel_m))
    return pc, pose


def split_scans(
    path: str | Path,
    out_dir: str | Path,
    voxel_m: float | None = 0.01,
    indices: list[int] | None = None,
) -> list[Path]:
    """Write ``<out_dir>/S<idx>.ply`` in SCANNER frame + ``S<idx>.pose.json``."""
    import json

    from minegs.ingest.e57.inventory import inventory

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = inventory(path)
    written = []
    for s in inv.scans:
        if indices is not None and s.index not in indices:
            continue
        pc, pose = read_scan(path, s.index, voxel_m)
        p = write_ply(pc, out_dir / f"S{s.index + 1:02d}.ply")
        (out_dir / f"S{s.index + 1:02d}.pose.json").write_text(
            json.dumps({"T_tls_from_scanner": pose.to_list(), "guid": s.guid}, indent=2)
        )
        written.append(p)
    return written
