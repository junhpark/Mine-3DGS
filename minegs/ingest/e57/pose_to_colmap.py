"""Scanner poses + ring crops -> COLMAP images/cameras in LOCAL_METRIC (§6.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minegs.core.frames import SE3
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec


@dataclass
class StationPose:
    station_id: str
    T_tls_from_scanner: SE3
    pano_id: str | None = None


def stations_to_colmap(
    stations: list[StationPose],
    spec: RingCropSpec,
    T_local_from_tls: SE3,
    image_ext: str = ".jpg",
    as_rig: bool = True,
) -> tuple[colmap_io.ColmapModel, dict[str, list[str]]]:
    """One PINHOLE camera (synthetic K) shared by every crop; one COLMAP rig per station if
    ``as_rig`` (all crops share the optical centre). Returns (model, station -> image names)."""
    cam = colmap_io.Camera.pinhole(1, spec.K(), spec.width, spec.height)
    model = colmap_io.ColmapModel({1: cam}, {}, {})
    members: dict[str, list[str]] = {}
    crops = spec.crops()
    if as_rig:
        sensors = [colmap_io.RigSensor(1, None)]
        model.rigs[1] = colmap_io.Rig(1, sensors)
    img_id = 1
    for fi, st in enumerate(stations, start=1):
        T_local_from_scanner = T_local_from_tls @ st.T_tls_from_scanner
        names = []
        ids = []
        for view in crops:
            name = f"{st.station_id}_{view.name}{image_ext}"
            T_local_from_cam = T_local_from_scanner @ SE3(view.R_scanner_from_cam, np.zeros(3))
            model.images[img_id] = colmap_io.Image.from_world_from_cam(
                img_id, T_local_from_cam, 1, name
            )
            names.append(name)
            ids.append((1, img_id))
            img_id += 1
        members[st.station_id] = names
        if as_rig:
            rig_from_world = T_local_from_scanner.inverse()
            model.frames[fi] = colmap_io.Frame(fi, 1, rig_from_world, ids)
    return model, members
