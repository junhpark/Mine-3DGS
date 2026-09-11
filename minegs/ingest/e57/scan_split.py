"""Per-station scan extraction (pye57): scanner-frame points + the scanner pose.

Spherical-only scans are converted to cartesian in the scanner frame.

Full extraction (tiling, downsampling policy, chunked writes) is Phase 0B.3. What this module
must already honour is the Phase 0B.1 pose contract: it reads poses through
``inventory.scan_pose`` and refuses scans whose pose is missing, unreadable or invalid, so the
same file can never be reported as "no usable pose" by the inventory and written out with an
identity pose by the splitter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minegs.core.pointcloud import PointCloud, voxel_downsample, write_ply
from minegs.ingest.common.geometry import spherical_to_cart
from minegs.ingest.e57.exceptions import E57PoseUnusableError
from minegs.ingest.e57.inventory import _pye57, scan_pose
from minegs.ingest.e57.models import ScanPose


def read_scan(
    path: str | Path, index: int, voxel_m: float | None = None
) -> tuple[PointCloud, ScanPose]:
    """Return (cloud in the SCANNER frame, its ``ScanPose``).

    The pose carries ``T_source_from_scanner`` — the E57's own SOURCE frame, not TLS_GLOBAL;
    that declaration belongs to dataset materialisation (Phase 0C). A scan whose pose is
    missing, unreadable or invalid raises ``E57PoseUnusableError`` rather than yielding a
    plausible identity.
    """
    pye57 = _pye57()
    e57 = pye57.E57(str(path))
    try:
        h = e57.get_header(index)
        pose, status = scan_pose(h)
        if pose is None or not pose.validation.valid:
            raise E57PoseUnusableError(
                f"scan index {index} has pose status {status!r}"
                + (f" ({'; '.join(pose.validation.issues)})" if pose is not None else "")
            )
        data = e57.read_scan_raw(index)
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
    """Write ``<out_dir>/<scan_id>.ply`` in the SCANNER frame + ``<scan_id>.pose.json``.

    Refuses the whole run if any selected scan's pose is unusable, naming the scans and
    pointing at the inventory. Extraction that tolerates unregistered scans is a Phase 0B.3
    decision with its own contract, not a default.
    """
    import json

    from minegs.ingest.e57.inventory import inventory

    out_dir = Path(out_dir)
    inv = inventory(path, compute_hash=False)
    selected = [s for s in inv.scans if indices is None or s.scan_index in indices]
    if not selected:
        raise E57PoseUnusableError(f"no scans selected from {path}")
    unusable = [(s.scan_id, s.pose_status) for s in selected if not s.pose_is_usable]
    if unusable:
        detail = ", ".join(f"{sid} ({status})" for sid, status in unusable)
        raise E57PoseUnusableError(
            f"{len(unusable)} of {len(selected)} selected scans have no usable pose: {detail}. "
            "Run `minegs ingest e57 inventory` to see why"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in selected:
        pc, pose = read_scan(path, s.scan_index, voxel_m)
        p = write_ply(pc, out_dir / f"{s.scan_id}.ply")
        # The E57's own coordinates are SOURCE until a later phase declares TLS_GLOBAL (§3).
        (out_dir / f"{s.scan_id}.pose.json").write_text(
            json.dumps(
                {
                    "scan_id": s.scan_id,
                    "scan_index": s.scan_index,
                    "guid": s.guid,
                    "source_frame": pose.source_frame,
                    "T_source_from_scanner": pose.T_source_from_scan,
                },
                indent=2,
            )
        )
        written.append(p)
    return written
