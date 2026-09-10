"""E57 inventory via pye57 (header only, no point payload). PDAL's readers.e57 merges scans and
drops spherical data, so scan enumeration / per-station split always goes through pye57."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import MissingDependencyError
from minegs.core.frames import SE3


class ScanInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    guid: str = ""
    name: str = ""
    point_count: int = 0
    has_cartesian: bool = False
    has_spherical: bool = False
    has_color: bool = False
    has_intensity: bool = False
    pose_tls_from_scanner: list[list[float]] | None = None
    bounds: dict[str, float] | None = None


class Image2DInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    guid: str = ""
    name: str = ""
    associated_scan_guid: str | None = None
    representation: str = "unknown"  # spherical | pinhole | cylindrical
    width: int | None = None
    height: int | None = None
    pose_tls_from_camera: list[list[float]] | None = None


class E57Inventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    scans: list[ScanInfo] = Field(default_factory=list)
    images: list[Image2DInfo] = Field(default_factory=list)
    coordinate_metadata: str | None = None

    def station_pano_map(self) -> dict[int, int | None]:
        """scan index -> image index (by associated GUID), None when the file has no pano."""
        by_guid = {
            im.associated_scan_guid: im.index for im in self.images if im.associated_scan_guid
        }
        return {s.index: by_guid.get(s.guid) for s in self.scans}


def _pye57():
    try:
        import pye57
    except ImportError as e:
        raise MissingDependencyError("pye57", "e57", "E57 inventory") from e
    return pye57


def _pose_from_header(header: Any) -> list[list[float]] | None:
    try:
        R = np.asarray(header.rotation_matrix, dtype=np.float64)
        t = np.asarray(header.translation, dtype=np.float64)
        return SE3(R, t).to_list()
    except Exception:
        return None


def inventory(path: str | Path) -> E57Inventory:
    pye57 = _pye57()
    path = Path(path)
    e57 = pye57.E57(str(path))
    inv = E57Inventory(file=str(path))
    try:
        for i in range(e57.scan_count):
            h = e57.get_header(i)
            fields = set(getattr(h, "point_fields", []) or [])
            inv.scans.append(
                ScanInfo(
                    index=i,
                    guid=str(getattr(h, "guid", "") or ""),
                    name=str(getattr(h, "name", "") or ""),
                    point_count=int(getattr(h, "point_count", 0) or 0),
                    has_cartesian="cartesianX" in fields,
                    has_spherical="sphericalRange" in fields,
                    has_color="colorRed" in fields,
                    has_intensity="intensity" in fields,
                    pose_tls_from_scanner=_pose_from_header(h),
                )
            )
        inv.images = _images2d(e57)
        root = e57.image_file.root()
        if root.isDefined("coordinateMetadata"):
            inv.coordinate_metadata = str(root["coordinateMetadata"].value())
    finally:
        e57.close()
    return inv


def _images2d(e57: Any) -> list[Image2DInfo]:
    """Walk ``/images2D`` with the raw libe57 node API; tolerate files without it."""
    out: list[Image2DInfo] = []
    try:
        root = e57.image_file.root()
        if not root.isDefined("images2D"):
            return out
        images = root["images2D"]
        for i in range(images.childCount()):
            node = images.get(i)
            rep, w, h = "unknown", None, None
            for key in (
                "sphericalRepresentation",
                "pinholeRepresentation",
                "cylindricalRepresentation",
            ):
                if node.isDefined(key):
                    rep = key.replace("Representation", "")
                    r = node[key]
                    w = int(r["imageWidth"].value()) if r.isDefined("imageWidth") else None
                    h = int(r["imageHeight"].value()) if r.isDefined("imageHeight") else None
                    break
            pose = None
            if node.isDefined("pose"):
                p = node["pose"]
                q = [p["rotation"][k].value() for k in ("w", "x", "y", "z")]
                t = [p["translation"][k].value() for k in ("x", "y", "z")]
                pose = SE3.from_quat_t(q, t).to_list()
            out.append(
                Image2DInfo(
                    index=i,
                    guid=str(node["guid"].value()) if node.isDefined("guid") else "",
                    name=str(node["name"].value()) if node.isDefined("name") else "",
                    associated_scan_guid=str(node["associatedData3DGuid"].value())
                    if node.isDefined("associatedData3DGuid")
                    else None,
                    representation=rep,
                    width=w,
                    height=h,
                    pose_tls_from_camera=pose,
                )
            )
    except Exception:
        return out
    return out
