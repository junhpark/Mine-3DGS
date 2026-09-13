"""Synthetic Phase 0B.3 staging tree — the G1 fixture for Phase 0C (§35).

Where ``minegs.core.synthetic`` writes a finished *dataset*, this writes what the E57
extractor would have written for a synthetic survey: registered SOURCE-frame scans with RGB,
their poses, embedded-style images with E57 pinhole (or spherical) metadata, and the three
0B artifacts through their real pydantic contracts. The dataset builder then has to earn
every frame and camera conversion the same way it would on a real tree.

The scene is honest about the things the Golden Gate must catch: the E57 image frame is
related to the COLMAP camera by a *known* axis convention (``spec.R_e57cam_from_cam``,
Matterport-like by default), image names carry a cube-face number that means nothing, and
the wall colour varies asymmetrically in every axis so a wrong rotation projects points onto
pixels of the wrong colour.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image as PILImage

from minegs.core.centerline import Centerline
from minegs.core.frames import SE3, rotmat_to_quat
from minegs.core.pointcloud import PointCloud, write_ply
from minegs.core.provenance import ProvenanceRecord, SourceAsset, git_commit, sha256_file
from minegs.core.provenance import tool_versions as _tool_versions
from minegs.ingest.common import colmap_io
from minegs.ingest.common.geometry import PanoConvention, dirs_to_equirect_uv
from minegs.ingest.e57.extract import E57ExtractionManifest, ImageOutput, ScanOutput
from minegs.ingest.e57.images import ImageAsset, image_id_for
from minegs.ingest.e57.mapping import MappingRecord, PanoMappingReport
from minegs.ingest.e57.models import (
    E57FileInfo,
    E57Inventory,
    E57ScanInventory,
    ImageSummary,
    StationCandidate,
    make_scan_pose,
    scan_id_for,
    station_id_for,
)

#: The SOURCE frame carries a survey-scale offset on purpose (float32-hostile, §3).
SOURCE_OFFSET = np.array([420150.0, 3961420.0, 85.0])

#: Matterport-like convention (PR #3 evidence): camera x = E57 +X, y = −Y, z = −Z.
MATTERPORT_LIKE = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))

ImageMode = Literal["pinhole_cube", "spherical"]


@dataclass
class StagingSpec:
    length_m: float = 90.0
    radius_m: float = 2.5
    curvature_deg_per_m: float = 0.4
    grade: float = 0.02
    station_spacing_m: float = 15.0
    points_per_m: int = 6000
    noise_m: float = 0.004
    image_size: int = 96
    image_mode: ImageMode = "pinhole_cube"
    n_faces: int = 6
    #: Which cube faces each station emits, in emission order. ``None`` means ``range(n_faces)``
    #: — the natural order, in which a face's *emission index* happens to equal its face number.
    #: Permuting it breaks that coincidence: only the pose metadata still says where a camera
    #: looks, so a builder that read orientation off ``source_index`` produces wrong poses.
    face_order: tuple[int, ...] | None = None
    R_e57cam_from_cam: tuple[tuple[float, float, float], ...] = MATTERPORT_LIKE
    pano_convention: PanoConvention = field(default_factory=PanoConvention)
    pixel_width_m: float = 2.0e-6  # focalLength = fx * pixel_width (a real-looking sensor)
    seed: int = 0
    file_name: str = "synthetic_survey.e57"


@dataclass
class StagingResult:
    root: Path
    staging_dir: Path
    source_placeholder: Path
    centerline_source: Centerline
    points_source: PointCloud
    point_chainage: np.ndarray
    station_poses: dict[str, SE3]  # station_id -> T_source_from_scanner
    image_poses_cam: dict[str, SE3]  # image_id -> T_source_from_cam (COLMAP camera)
    R_e57cam_from_cam: np.ndarray
    spec: StagingSpec


# ------------------------------------------------------------------------------ scene


def make_centerline(spec: StagingSpec, step_m: float = 1.0) -> Centerline:
    n = int(spec.length_m / step_m) + 1
    s = np.arange(n) * step_m
    heading = np.radians(spec.curvature_deg_per_m) * s
    x = np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * step_m)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(heading[:-1]) * step_m)])
    z = spec.grade * s
    return Centerline(np.column_stack([x, y, z]) + SOURCE_OFFSET, "SOURCE", "design")


def wall_colour(xyz_source: np.ndarray) -> np.ndarray:
    """Asymmetric in every axis: no axis flip or 90° turn maps the pattern onto itself."""
    p = np.asarray(xyz_source, dtype=np.float64) - SOURCE_OFFSET
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = 128 + 100 * np.sin(0.9 * x + 0.3 * z)
    g = 128 + 100 * np.sin(1.3 * y + 0.7 + 0.2 * x)
    b = 128 + 100 * np.cos(0.6 * z + 1.1 * x - 0.4 * y)
    return np.clip(np.column_stack([r, g, b]), 0, 255).astype(np.uint8)


def sample_wall(cl: Centerline, spec: StagingSpec, rng: np.random.Generator):
    n = int(spec.length_m * spec.points_per_m)
    s = rng.uniform(cl.s_start, cl.s_end, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    R, origin = cl.frames_at(s)
    local = np.column_stack(
        [np.zeros(n), spec.radius_m * np.cos(theta), spec.radius_m * np.sin(theta)]
    )
    pts = origin + np.einsum("nij,nj->ni", R, local)
    pts += rng.normal(0, spec.noise_m, pts.shape)
    return pts, s


def cube_face_rotations() -> list[np.ndarray]:
    """``R_scanner_from_cam`` for six faces: forward, left, back, right, up, down.

    Scanner frame: x tangent, y left, z up. COLMAP camera: x right, y down, z forward.
    """
    fwd = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    down = [(0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]
    out = []
    for f, d in zip(fwd, down, strict=True):
        z = np.array(f, dtype=np.float64)
        y = np.array(d, dtype=np.float64)
        x = np.cross(y, z)
        out.append(np.column_stack([x, y, z]))
    return out


def _face_plan(spec: StagingSpec) -> list[tuple[int, np.ndarray]]:
    """``(face_number, R_scanner_from_cam)`` per image of a station, in emission order."""
    all_faces = cube_face_rotations()
    order = tuple(range(spec.n_faces)) if spec.face_order is None else tuple(spec.face_order)
    if len(order) != spec.n_faces or len(set(order)) != len(order):
        raise ValueError(f"face_order {order} is not {spec.n_faces} distinct faces")
    if not all(0 <= f < len(all_faces) for f in order):
        raise ValueError(f"face_order {order} names a face outside 0..{len(all_faces) - 1}")
    return [(f, all_faces[f]) for f in order]


def render_pinhole(xyz_cam: np.ndarray, rgb: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    z = xyz_cam[:, 2]
    m = z > 0.05
    if not m.any():
        return img
    uv = (xyz_cam[m, :2] / z[m, None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    u = uv[inside, 0].astype(int)
    v = uv[inside, 1].astype(int)
    c = rgb[m][inside]
    order = np.argsort(-z[m][inside])  # far first, near paints last
    u, v, c = u[order], v[order], c[order]
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            img[np.clip(v + dv, 0, size - 1), np.clip(u + du, 0, size - 1)] = c
    return img


def render_equirect(
    xyz_scanner: np.ndarray, rgb: np.ndarray, width: int, height: int, conv: PanoConvention
) -> np.ndarray:
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    r = np.linalg.norm(xyz_scanner, axis=1)
    uv = dirs_to_equirect_uv(xyz_scanner, width, height, conv)
    u = np.mod(uv[:, 0].astype(int), width)
    v = np.clip(uv[:, 1].astype(int), 0, height - 1)
    order = np.argsort(-r)
    u, v, c = u[order], v[order], rgb[order]
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            img[np.clip(v + dv, 0, height - 1), np.mod(u + du, width)] = c
    return img


# ------------------------------------------------------------------------------ writer


def generate_staging(root: str | Path, spec: StagingSpec | None = None) -> StagingResult:
    spec = spec or StagingSpec()
    rng = np.random.default_rng(spec.seed)
    root = Path(root)
    staging = root / "staging"
    raw = root / "raw"
    for d in (staging / "scans", staging / "images", raw):
        d.mkdir(parents=True, exist_ok=True)

    # A placeholder for the E57 whose bytes the artifacts hash: provenance stays a chain of
    # real digests even though no libE57 file exists here.
    src = raw / spec.file_name
    src.write_bytes(hashlib.sha256(f"synthetic staging seed={spec.seed}".encode()).digest())
    src_sha = sha256_file(src)
    R_conv = np.asarray(spec.R_e57cam_from_cam, dtype=np.float64)

    cl = make_centerline(spec)
    pts, s_pts = sample_wall(cl, spec, rng)
    rgb = wall_colour(pts)
    station_s = np.arange(spec.station_spacing_m / 2, spec.length_m, spec.station_spacing_m)
    reach = spec.station_spacing_m

    prov = ProvenanceRecord(
        git_commit=git_commit(),
        source_assets=[SourceAsset(path=str(src), sha256=src_sha, size_bytes=src.stat().st_size)],
        tool_versions=_tool_versions(),
    )
    scans_inv: list[E57ScanInventory] = []
    stations: list[StationCandidate] = []
    scan_outputs: list[ScanOutput] = []
    assets: list[ImageAsset] = []
    mappings: list[MappingRecord] = []
    image_outputs: list[ImageOutput] = []
    station_poses: dict[str, SE3] = {}
    image_poses: dict[str, SE3] = {}
    img_index = 0
    faces = _face_plan(spec)
    K = None
    if spec.image_mode == "pinhole_cube":
        f_px = spec.image_size / 2.0  # 90° face
        K = np.array([[f_px, 0, spec.image_size / 2], [0, f_px, spec.image_size / 2], [0, 0, 1.0]])

    for si, s in enumerate(station_s):
        scan_id, station_id = scan_id_for(si), station_id_for(si)
        guid = f"{{scan-{si:04d}-guid}}"
        T_source_from_scanner = SE3(cl.frame_at(float(s)).R, cl.point_at(float(s)))
        station_poses[station_id] = T_source_from_scanner
        q = rotmat_to_quat(T_source_from_scanner.R)
        pose = make_scan_pose(T_source_from_scanner.R, T_source_from_scanner.t, q)

        near = np.abs(s_pts - s) <= reach
        cloud = PointCloud(pts[near], rgb[near], frame="SOURCE")
        ply = write_ply(cloud, staging / "scans" / f"{scan_id}.ply", xyz_dtype="f8")
        (staging / "scans" / f"{scan_id}.pose.json").write_text(
            _pose_json(scan_id, si, guid, pose.T_source_from_scan)
        )
        scans_inv.append(
            E57ScanInventory(
                scan_index=si,
                scan_id=scan_id,
                name=f"Sweep {si}",
                guid=guid,
                point_count=int(near.sum()),
                has_cartesian_xyz=True,
                has_rgb=True,
                pose_declared=True,
                pose_status="valid",
                pose=pose,
                raw_point_fields=[
                    "cartesianX",
                    "cartesianY",
                    "cartesianZ",
                    "colorRed",
                    "colorGreen",
                    "colorBlue",
                ],
            )
        )
        stations.append(StationCandidate(station_id=station_id, scan_ids=[scan_id]))
        scan_outputs.append(
            ScanOutput(
                scan_id=scan_id,
                scan_index=si,
                path=str(ply.resolve()),
                point_count_input=int(near.sum()),
                point_count_masked=int(near.sum()),
                point_count_output=int(near.sum()),
                source_frame="SOURCE",
                registration_status="registered",
                pose_status="valid",
                pose=pose,
                attributes=["red", "green", "blue"],
                sha256=sha256_file(ply),
            )
        )

        # images for this station
        if spec.image_mode == "pinhole_cube":
            assert K is not None
            for face, R_scanner_from_cam in faces:
                image_id = image_id_for(img_index)
                T_source_from_cam = T_source_from_scanner @ SE3(R_scanner_from_cam, np.zeros(3))
                # E57 stores the camera in *its* image frame: T_source_from_e57cam
                T_source_from_e57cam = T_source_from_cam @ SE3(R_conv.T, np.zeros(3))
                xyz_cam = T_source_from_cam.inverse().apply(pts[near])
                img = render_pinhole(xyz_cam, rgb[near], K, spec.image_size)
                path = staging / "images" / f"{image_id}.png"
                PILImage.fromarray(img).save(path)
                image_poses[image_id] = T_source_from_cam
                meta = {
                    "focalLength": float(K[0, 0] * spec.pixel_width_m),
                    "pixelWidth": spec.pixel_width_m,
                    "pixelHeight": spec.pixel_width_m,
                    "principalPointX": float(K[0, 2]),
                    "principalPointY": float(K[1, 2]),
                    "pose_rotation_wxyz": rotmat_to_quat(T_source_from_e57cam.R).tolist(),
                    "pose_translation": T_source_from_e57cam.t.tolist(),
                }
                _append_image(
                    assets,
                    mappings,
                    image_outputs,
                    image_id,
                    img_index,
                    guid,
                    scan_id,
                    station_id,
                    "pinhole",
                    "pinholeRepresentation",
                    spec.image_size,
                    spec.image_size,
                    path,
                    meta,
                    name=f"Skybox {face}",
                )
                img_index += 1
        else:
            image_id = image_id_for(img_index)
            W, H = 4 * spec.image_size, 2 * spec.image_size
            xyz_sc = T_source_from_scanner.inverse().apply(pts[near])
            img = render_equirect(xyz_sc, rgb[near], W, H, spec.pano_convention)
            path = staging / "images" / f"{image_id}.png"
            PILImage.fromarray(img).save(path)
            image_poses[image_id] = T_source_from_scanner
            meta = {
                "pose_rotation_wxyz": q.tolist(),
                "pose_translation": T_source_from_scanner.t.tolist(),
            }
            _append_image(
                assets,
                mappings,
                image_outputs,
                image_id,
                img_index,
                guid,
                scan_id,
                station_id,
                "spherical",
                "sphericalRepresentation",
                W,
                H,
                path,
                meta,
                name=f"Pano {si}",
            )
            img_index += 1

    inv = E57Inventory(
        file=E57FileInfo(
            path=str(src),
            file_name=src.name,
            size_bytes=src.stat().st_size,
            sha256=src_sha,
            format_name="synthetic (no libE57 file)",
        ),
        scan_count=len(scans_inv),
        scans=scans_inv,
        station_candidates=stations,
        images=ImageSummary(has_images2d=True, image_count=img_index, enumeration_status="ok"),
        notes=["synthetic staging tree written by minegs.core.synthetic_staging"],
        provenance=prov,
    )
    inv.save(staging / "inventory.json")
    report = PanoMappingReport(
        source_file=str(src),
        source_sha256=src_sha,
        scan_count=len(scans_inv),
        images=assets,
        mappings=mappings,
        provenance=prov,
    )
    report.save(staging / "pano_mapping.json")
    manifest = E57ExtractionManifest(
        source_e57=str(src),
        source_sha256=src_sha,
        work_dir=str(staging.resolve()),
        registration="registered",
        output_frame="SOURCE",
        scan_outputs=scan_outputs,
        image_outputs=image_outputs,
        mapping_report=report,
        notes=["synthetic staging tree; the source E57 is a placeholder whose bytes are hashed"],
        provenance=prov,
    )
    manifest.save(staging / "extraction_manifest.json")
    return StagingResult(
        root=root,
        staging_dir=staging,
        source_placeholder=src,
        centerline_source=cl,
        points_source=PointCloud(pts, rgb, frame="SOURCE"),
        point_chainage=s_pts,
        station_poses=station_poses,
        image_poses_cam=image_poses,
        R_e57cam_from_cam=R_conv,
        spec=spec,
    )


def _append_image(
    assets,
    mappings,
    outputs,
    image_id,
    index,
    guid,
    scan_id,
    station_id,
    rep,
    rep_node,
    width,
    height,
    path: Path,
    meta,
    name,
) -> None:
    nbytes = path.stat().st_size
    assets.append(
        ImageAsset(
            image_id=image_id,
            source="e57_embedded",
            source_index=index,
            guid=f"{{image-{index:04d}-guid}}",
            name=name,
            associated_scan_guid=guid,
            representation=rep,
            representation_source=rep_node,
            width=width,
            height=height,
            blob_field="pngImage",
            blob_bytes=nbytes,
            vendor_metadata=meta,
        )
    )
    mappings.append(
        MappingRecord(
            image_id=image_id,
            scan_id=scan_id,
            station_id=station_id,
            status="confirmed",
            evidence_type="e57_associated_guid",
            evidence_value=guid,
            reason="associatedData3DGuid names exactly one scan",
        )
    )
    outputs.append(
        ImageOutput(
            image_id=image_id,
            path=str(path.resolve()),
            source="e57_embedded",
            extracted=True,
            representation=rep,
            width=width,
            height=height,
            image_format="png",
            blob_field="pngImage",
            bytes_written=nbytes,
            mapped_scan_id=scan_id,
            mapped_station_id=station_id,
            mapping_status="confirmed",
            mapping_evidence_type="e57_associated_guid",
            sha256=sha256_file(path),
        )
    )


def _pose_json(scan_id: str, index: int, guid: str, T: list[list[float]]) -> str:
    import json

    return (
        json.dumps(
            {
                "scan_id": scan_id,
                "scan_index": index,
                "guid": guid,
                "name": f"Sweep {index}",
                "point_frame": "SOURCE",
                "source_frame": "SOURCE",
                "pose_status": "valid",
                "registration_status": "registered",
                "T_source_from_scanner": T,
            },
            indent=2,
        )
        + "\n"
    )


__all__ = ["StagingResult", "StagingSpec", "colmap_io", "generate_staging"]
