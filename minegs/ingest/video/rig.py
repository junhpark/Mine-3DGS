"""360 crops of one frame -> COLMAP rig (§6.2). Same maths as the E57 ring crop; here the
frame's pose is unknown (SfM estimates rig_from_world) but the sensor_from_rig extrinsics
are exact because K and the crop rotations are synthetic. Emits COLMAP's
``rig_config.json`` (for ``colmap rig_configurator``) and our own ``Rig`` objects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minegs.core.frames import SE3
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec


def rig_from_ring(spec: RingCropSpec, camera_ids: list[int] | None = None) -> colmap_io.Rig:
    crops = spec.crops()
    cam_ids = camera_ids or list(range(1, len(crops) + 1))
    ref = crops[0]
    T_scanner_from_ref = SE3(ref.R_scanner_from_cam, np.zeros(3))
    sensors = [colmap_io.RigSensor(cam_ids[0], None)]
    for view, cid in zip(crops[1:], cam_ids[1:], strict=True):
        T_scanner_from_cam = SE3(view.R_scanner_from_cam, np.zeros(3))
        cam_from_ref = T_scanner_from_cam.inverse() @ T_scanner_from_ref
        sensors.append(colmap_io.RigSensor(cid, cam_from_ref))
    return colmap_io.Rig(1, sensors)


def rig_config_json(spec: RingCropSpec, image_prefixes: list[str] | None = None) -> list[dict]:
    """COLMAP ``rig_config.json``: one rig, one camera per crop, cam_from_rig from the ring."""
    crops = spec.crops()
    prefixes = image_prefixes or [f"{v.name}/" for v in crops]
    rig = rig_from_ring(spec)
    cams = []
    for view, prefix, sensor in zip(crops, prefixes, rig.sensors, strict=True):
        entry: dict = {"image_prefix": prefix}
        if sensor.sensor_from_rig is None:
            entry["ref_sensor"] = True
        else:
            entry["cam_from_rig_rotation"] = sensor.sensor_from_rig.quat().tolist()
            entry["cam_from_rig_translation"] = sensor.sensor_from_rig.t.tolist()
        K = spec.K()
        entry["camera_model_name"] = "PINHOLE"
        entry["camera_params"] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        del view
        cams.append(entry)
    return [{"cameras": cams}]


def write_rig_config(
    spec: RingCropSpec, path: str | Path, image_prefixes: list[str] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rig_config_json(spec, image_prefixes), indent=2))
    return path
