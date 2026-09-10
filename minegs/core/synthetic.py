"""Synthetic tunnel dataset — the Phase 0A gate (§13): a dataset that satisfies the
contract end-to-end without any real E57/video, so manifest, frames, chunking, COLMAP
export, init PLY, protocol inference and the geometry/section/volume evaluators can all be
exercised in CI.

Scene: a gently curving drift of radius ``radius_m`` in TLS_GLOBAL with a UTM-like offset
(to prove the float32 argument of §3), TLS stations every ``station_spacing_m`` with a ring
of pinhole crops, plus one video trajectory segment. Test groups and a chainage holdout are
declared so ``geometry_holdout`` / ``novel_view`` protocols are both available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import minegs
from minegs.core.centerline import Centerline
from minegs.core.chunking import assign_groups, plan_chunks
from minegs.core.frames import SE3
from minegs.core.manifest import (
    CaptureEpoch,
    CaptureGroup,
    CenterlineRef,
    CoordinateFrames,
    GeometryHoldout,
    Initialization,
    Manifest,
    ManifestProvenance,
    PanoConvention,
    Scale,
    Split,
)
from minegs.core.pointcloud import PointCloud, write_ply
from minegs.core.provenance import git_commit, tool_versions
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec

TLS_OFFSET = np.array([318300.0, 4012300.0, 120.0])  # UTM-ish, deliberately float32-hostile


@dataclass
class SyntheticSpec:
    dataset_id: str = "synthetic_tunnel_ep1_v001"
    length_m: float = 120.0
    radius_m: float = 2.5
    curvature_deg_per_m: float = 0.4
    grade: float = 0.02  # rise per m
    station_spacing_m: float = 15.0
    points_per_m: int = 1500
    noise_m: float = 0.005
    n_yaw: int = 6
    image_size: int = 96
    test_every: int = 4  # every 4th station is a render test group
    holdout_ranges_m: tuple[tuple[float, float], ...] = ((38.0, 46.0),)
    chunk_length_m: float = 80.0
    chunk_overlap_m: float = 15.0
    seed: int = 0
    with_video_segment: bool = True


@dataclass
class SyntheticResult:
    root: Path
    dataset_dir: Path
    manifest: Manifest
    centerline_tls: Centerline
    tls_points_global: PointCloud


def make_centerline(spec: SyntheticSpec, step_m: float = 1.0) -> Centerline:
    n = int(spec.length_m / step_m) + 1
    s = np.arange(n) * step_m
    heading = np.radians(spec.curvature_deg_per_m) * s
    x = np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * step_m)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(heading[:-1]) * step_m)])
    z = spec.grade * s
    return Centerline(np.column_stack([x, y, z]) + TLS_OFFSET, "TLS_GLOBAL", "design")


def sample_tunnel_surface(
    cl: Centerline, spec: SyntheticSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Points on the tunnel wall in TLS_GLOBAL, plus their chainage."""
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


def _render_station_image(
    xyz_cam: np.ndarray, K: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Tiny fake photo: depth-shaded splats of visible points. Enough for layout + smoke tests."""
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    z = xyz_cam[:, 2]
    m = z > 0.1
    if m.any():
        uv = (xyz_cam[m, :2] / z[m, None]) * np.array([K[0, 0], K[1, 1]]) + np.array(
            [K[0, 2], K[1, 2]]
        )
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
        uv = uv[inside].astype(int)
        shade = np.clip(255 - 18 * z[m][inside], 30, 255).astype(np.uint8)
        img[uv[:, 1], uv[:, 0]] = shade[:, None]
    img = np.clip(img.astype(int) + rng.integers(-5, 5, img.shape), 0, 255).astype(np.uint8)
    return img


def generate(root: str | Path, spec: SyntheticSpec | None = None) -> SyntheticResult:
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    root = Path(root)
    raw = root / "raw"
    ds = root / "dataset"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    cl_tls = make_centerline(spec)
    cl_tls.to_csv(raw / "centerline.csv")
    pts_tls, s_pts = sample_tunnel_surface(cl_tls, spec, rng)
    rgb = np.clip(rng.normal(120, 20, pts_tls.shape), 0, 255).astype(np.uint8)
    tls_cloud = PointCloud(pts_tls, rgb, frame="TLS_GLOBAL")
    write_ply(tls_cloud, raw / "tls_full.ply", xyz_dtype="f8")

    # LOCAL_METRIC origin = tunnel centroid (rounded to keep the transform readable)
    origin = np.round(cl_tls.point_at(0.5 * (cl_tls.s_start + cl_tls.s_end)), 1)
    T_tls_from_local = SE3.from_translation(origin)
    T_local_from_tls = T_tls_from_local.inverse()
    cl_local = cl_tls.transformed(T_local_from_tls, "LOCAL_METRIC")

    # stations + ring crops
    crop = RingCropSpec(
        n_yaw=spec.n_yaw, fov_deg=90.0, width=spec.image_size, height=spec.image_size
    )
    K = crop.K()
    cameras = {1: colmap_io.Camera.pinhole(1, K, crop.width, crop.height)}
    images: dict[int, colmap_io.Image] = {}
    groups: dict[str, CaptureGroup] = {}
    station_s = np.arange(spec.station_spacing_m / 2, spec.length_m, spec.station_spacing_m)
    img_id = 1
    pts_local = T_local_from_tls.apply(pts_tls)
    for si, s in enumerate(station_s):
        sid = f"S{si + 1:02d}"
        T_local_from_scanner = SE3(cl_local.frame_at(float(s)).R, cl_local.point_at(float(s)))
        members = []
        for view in crop.crops():
            name = f"{sid}_{view.name}.png"
            T_local_from_cam = T_local_from_scanner @ SE3(view.R_scanner_from_cam, np.zeros(3))
            im = colmap_io.Image.from_world_from_cam(img_id, T_local_from_cam, 1, name)
            images[img_id] = im
            near = np.abs(s_pts - s) < 12.0
            xyz_cam = im.cam_from_world.apply(pts_local[near])
            PILImage.fromarray(_render_station_image(xyz_cam, K, spec.image_size, rng)).save(
                ds / "images" / name
            )
            members.append(name)
            img_id += 1
        groups[sid] = CaptureGroup(
            type="tls_station",
            members=members,
            chainage_m=round(float(s), 2),
            pano_id=f"pano_{sid}",
        )

    if spec.with_video_segment:
        vid = "V001"
        members = []
        s_lo, s_hi = 0.25 * spec.length_m, 0.55 * spec.length_m
        for k, s in enumerate(np.arange(s_lo, s_hi, 2.0)):
            name = f"v_{k:06d}.png"
            T_local_from_cam = SE3(
                cl_local.frame_at(float(s)).R,
                cl_local.point_at(float(s)) + np.array([0.0, 0.0, -0.8]),
            ) @ SE3(crop.crops()[0].R_scanner_from_cam, np.zeros(3))
            im = colmap_io.Image.from_world_from_cam(img_id, T_local_from_cam, 1, name)
            images[img_id] = im
            near = np.abs(s_pts - s) < 12.0
            PILImage.fromarray(
                _render_station_image(
                    im.cam_from_world.apply(pts_local[near]), K, spec.image_size, rng
                )
            ).save(ds / "images" / name)
            members.append(name)
            img_id += 1
        groups[vid] = CaptureGroup(
            type="trajectory_segment",
            members=members,
            chainage_range_m=(round(s_lo, 2), round(s_hi, 2)),
        )

    # split: every Nth station -> test; chainage holdout excluded from init points
    station_ids = [g for g in groups if g.startswith("S")]
    test_groups = [g for i, g in enumerate(station_ids) if (i + 1) % spec.test_every == 0]
    train_groups = [g for g in groups if g not in test_groups]
    holdout = [tuple(r) for r in spec.holdout_ranges_m]
    excluded = np.zeros(len(pts_tls), dtype=bool)
    for lo, hi in holdout:
        excluded |= (s_pts >= lo) & (s_pts <= hi)
    # init points come from train TLS stations only (station reach = +- spacing)
    init_mask = ~excluded
    reach = np.zeros(len(pts_tls), dtype=bool)
    for g in train_groups:
        if groups[g].type == "tls_station":
            reach |= np.abs(s_pts - groups[g].chainage_m) <= spec.station_spacing_m
    init_mask &= reach
    init_idx = np.flatnonzero(init_mask)
    init_idx = rng.choice(init_idx, size=min(len(init_idx), 60_000), replace=False)
    init_cloud = PointCloud(pts_local[init_idx], rgb[init_idx], frame="LOCAL_METRIC")
    write_ply(init_cloud, ds / "init_points.ply")

    # sparse points: small subsample, no tracks (synthetic)
    sp_idx = rng.choice(len(pts_local), size=min(len(pts_local), 5000), replace=False)
    points3D = {
        int(i) + 1: colmap_io.Point3D(int(i) + 1, pts_local[j], rgb[j])
        for i, j in enumerate(sp_idx)
    }
    model = colmap_io.ColmapModel(cameras, images, points3D)
    colmap_io.write_model(model, ds / "sparse" / "0")
    cl_tls.to_csv(ds / "centerline.csv")

    chunks = assign_groups(
        plan_chunks(
            cl_tls.s_start, cl_tls.s_end, spec.chunk_length_m, spec.chunk_overlap_m, cl_tls
        ),
        {g: span for g, grp in groups.items() if (span := grp.span()) is not None},
    )
    init_groups = [g for g in train_groups if groups[g].type == "tls_station"]
    manifest = Manifest(
        dataset_id=spec.dataset_id,
        coordinate_frames=CoordinateFrames(T_tls_from_local=T_tls_from_local.to_list()),
        capture_groups=groups,
        split=Split(
            train_groups=train_groups,
            test_groups=test_groups,
            geometry_holdout=GeometryHoldout(
                chainage_ranges_m=holdout, points_excluded=True, images_excluded=False
            ),
        ),
        initialization=Initialization(
            source="tls",
            file="init_points.ply",
            groups=init_groups,
            excluded_chainage_ranges_m=holdout,
            n_points=len(init_cloud),
        ),
        provenance=ManifestProvenance(
            minegs_version=minegs.__version__,
            git_commit=git_commit(),
            config_hash="synthetic",
            tool_versions=tool_versions(),
        ),
        source="tls",
        capture_epoch=CaptureEpoch(id="ep1", date="2026-01-01"),
        scale=Scale(basis="tls_pose", factor=1.0),
        pano_convention=PanoConvention(source="E57Embedded", vendor="synthetic"),
        centerline=CenterlineRef(file="centerline.csv", source="design", frame="TLS_GLOBAL"),
        chunks=chunks,
    )
    manifest.save_dataset(ds)
    return SyntheticResult(root, ds, manifest, cl_tls, tls_cloud)
