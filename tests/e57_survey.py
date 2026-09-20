"""Write a genuine E57 for the synthetic survey, at test time (§Phase 2 C6).

Every other E57 in this suite is either a fake pye57 harness or a handful of points. Neither
one lets the end-to-end gate start where a real run starts: at a file libE57 wrote, carrying
registered scans *and* embedded ``/images2D`` entries with intrinsics and poses. Without it the
gate would have to begin at a staging tree, and the one link it could never test is the one
that decodes the survey.

So the synthetic staging tree is turned back into an E57 here: each station's cloud is taken
back into its own scanner frame and written with its pose, and each rendered face becomes a
``pinholeRepresentation`` with a ``pngImage`` blob, the same node shapes a vendor exporter
produces. ``extract`` then has to earn the staging tree again from the file, which is what the
gate needs it to do.

``pye57`` has no images2D writer, so the entries are built through the ``libe57`` node API --
the same way ``tests/test_e57_extract.py`` builds the Phase 0C G2 fixture. No E57 is ever
committed: this writes one into a temporary directory and it dies with the test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from minegs.core.frames import SE3, rotmat_to_quat
from minegs.core.pointcloud import read_ply

_INTRINSICS = ("focalLength", "pixelWidth", "pixelHeight", "principalPointX", "principalPointY")


def write_survey_e57(staging_dir: str | Path, path: str | Path) -> Path:
    """Serialise a Phase 0B.3 staging tree back into a real E57 file.

    The scans go in as libE57 wants them — points in the scanner's own frame, the registration
    carried by the pose — so that extraction has to apply the pose itself rather than being
    handed the answer.
    """
    import pye57
    from pye57 import libe57

    staging, path = Path(staging_dir), Path(path)
    mapping = json.loads((staging / "pano_mapping.json").read_text())
    images = {a["image_id"]: a for a in mapping["images"]}
    by_scan: dict[str, list[str]] = {}
    for rec in mapping["mappings"]:
        by_scan.setdefault(rec["scan_id"], []).append(rec["image_id"])

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = pye57.E57(str(path), mode="w")
    try:
        guids = _write_scans(handle, staging)
        _write_images(handle, libe57, staging, guids, by_scan, images)
    finally:
        handle.close()
    return path


def _write_scans(handle, staging: Path) -> list[tuple[str, str]]:
    """One scan per station, in SCANNER coordinates, with its pose. Returns libE57's own guids."""
    out: list[tuple[str, str]] = []
    for i, ply in enumerate(sorted((staging / "scans").glob("*.ply"))):
        scan_id = ply.stem
        pose = json.loads((staging / "scans" / f"{scan_id}.pose.json").read_text())
        T = np.asarray(pose["T_source_from_scanner"], dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        cloud = read_ply(ply)
        xyz = SE3(R, t).inverse().apply(cloud.xyz)
        rgb = cloud.rgb.astype(np.float64)
        handle.write_scan_raw(
            {
                "cartesianX": np.ascontiguousarray(xyz[:, 0]),
                "cartesianY": np.ascontiguousarray(xyz[:, 1]),
                "cartesianZ": np.ascontiguousarray(xyz[:, 2]),
                "colorRed": np.ascontiguousarray(rgb[:, 0]),
                "colorGreen": np.ascontiguousarray(rgb[:, 1]),
                "colorBlue": np.ascontiguousarray(rgb[:, 2]),
            },
            name=pose.get("name") or f"Sweep {i}",
            rotation=rotmat_to_quat(R),
            translation=t,
        )
        # The file's own guid, not the synthetic one: the images have to point at what libE57
        # actually wrote, which is the association evidence Phase 0B.2 maps on.
        out.append((scan_id, handle.get_header(i)["guid"].value()))
    return out


def _write_images(handle, libe57, staging: Path, guids, by_scan, images) -> None:
    imf = handle.image_file
    images2D = libe57.VectorNode(imf.root().get("images2D"))
    for scan_id, guid in guids:
        for image_id in by_scan.get(scan_id, []):
            asset = images[image_id]
            meta = asset["vendor_metadata"]
            payload = (staging / "images" / f"{image_id}.png").read_bytes()

            entry = libe57.StructureNode(imf)
            entry.set("guid", libe57.StringNode(imf, asset["guid"]))
            entry.set("name", libe57.StringNode(imf, asset["name"]))
            entry.set("associatedData3DGuid", libe57.StringNode(imf, guid))
            entry.set("pose", _pose_node(libe57, imf, meta))

            rep = libe57.StructureNode(imf)
            rep.set("imageWidth", libe57.IntegerNode(imf, int(asset["width"])))
            rep.set("imageHeight", libe57.IntegerNode(imf, int(asset["height"])))
            for field in _INTRINSICS:
                rep.set(field, libe57.FloatNode(imf, float(meta[field])))
            blob = libe57.BlobNode(imf, len(payload))
            rep.set("pngImage", blob)
            entry.set("pinholeRepresentation", rep)

            images2D.append(entry)  # attaches the blob to the tree...
            blob.write(bytearray(payload), 0, len(payload))  # ...so it can be written


def _pose_node(libe57, imf, meta: dict):
    pose = libe57.StructureNode(imf)
    rotation = libe57.StructureNode(imf)
    for key, value in zip("wxyz", meta["pose_rotation_wxyz"], strict=True):
        rotation.set(key, libe57.FloatNode(imf, float(value)))
    pose.set("rotation", rotation)
    translation = libe57.StructureNode(imf)
    for key, value in zip("xyz", meta["pose_translation"], strict=True):
        translation.set(key, libe57.FloatNode(imf, float(value)))
    pose.set("translation", translation)
    return pose
