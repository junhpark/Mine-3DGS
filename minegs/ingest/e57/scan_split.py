"""Per-station scanner-frame extraction — the Phase 0A flat layout.

Superseded by ``minegs ingest e57 extract`` (``minegs.ingest.e57.extract``), which writes the
Phase 0B.3 staging tree, supports registered output and records an extraction manifest. This
module is kept as the flat ``<out_dir>/<scan_id>.ply`` layout the Phase 0A dataset builder
reads, and now shares the extractor's payload reader so the two cannot disagree about the
same scan: in particular the invalid-state mask is applied to every attribute here too (it
used to slice colour to the coordinates' length, which shifts colour onto the wrong points).

It stays stricter than the new ``--raw`` path: a scan with no pose is refused, because this
layout carries no ``registration_status`` and an unregistered cloud sitting next to a
registered one is indistinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

from minegs.core.pointcloud import PointCloud, voxel_downsample
from minegs.ingest.e57 import _nodes
from minegs.ingest.e57.exceptions import E57PoseUnusableError
from minegs.ingest.e57.extract import SCANNER_FRAME, SOURCE_FRAME, read_scan_points
from minegs.ingest.e57.inventory import scan_pose
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
    with _nodes.open_e57(path) as e57:
        h = e57.get_header(index)
        pose, status = scan_pose(h)
        if pose is None or not pose.validation.valid:
            raise E57PoseUnusableError(
                f"scan index {index} has pose status {status!r}"
                + (f" ({'; '.join(pose.validation.issues)})" if pose is not None else "")
            )
        pc, _meta = read_scan_points(e57, index, h)
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
    pointing at the inventory. Extraction that tolerates unregistered scans is the Phase 0B.3
    ``--raw`` contract, which marks its output accordingly; it is not a default here.
    """
    from minegs.core.pointcloud import write_ply
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
        p = write_ply(pc, out_dir / f"{s.scan_id}.ply", xyz_dtype="f8")
        # The E57's own coordinates are SOURCE until a later phase declares TLS_GLOBAL (§3).
        (out_dir / f"{s.scan_id}.pose.json").write_text(
            json.dumps(
                {
                    "scan_id": s.scan_id,
                    "scan_index": s.scan_index,
                    "guid": s.guid,
                    "point_frame": SCANNER_FRAME,
                    "source_frame": SOURCE_FRAME,
                    "T_source_from_scanner": pose.T_source_from_scan,
                },
                indent=2,
            )
        )
        written.append(p)
    return written
