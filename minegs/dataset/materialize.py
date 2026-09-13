"""Materialise a Phase 0B staging tree into the ``dataset/`` contract (§3–§4, §14–§21, §27–§30).

The build is a chain of decisions, each recorded:

1. read the staging tree through its 0B contracts (``staging_input``);
2. declare SOURCE → TLS_GLOBAL and derive the LOCAL_METRIC origin (``frames``);
3. one capture group per TLS station, from *resolved* mappings only (§14);
4. COLMAP cameras and poses in LOCAL_METRIC from the E57's own intrinsics and poses and a
   measured or declared axis convention (``cameras``);
5. optional centerline, split and chainage holdout (§17–§19);
6. ``init_points.ply`` from the initialization groups' scans with every point inside a
   holdout range removed *by its own coordinates* (§16) — and ``points3D.txt`` from the same
   set, so no second path reintroduces the geometry (§20);
7. manifest + provenance naming every input (§21);
8. §27 numerical sanity, ``Manifest.load_dataset``, ``consistency_issues`` and the protocol
   judgement, all before anything is published (§29–§30).

Nothing is written into the final directory until all of that has passed.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import minegs
from minegs.core.centerline import Centerline
from minegs.core.config import config_hash
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, check_float32_safe
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
from minegs.core.pointcloud import PointCloud, read_ply, voxel_downsample, write_ply
from minegs.core.provenance import SourceAsset, git_commit, sha256_file, tool_versions
from minegs.dataset.build_config import DatasetBuildConfig, PanoConventionConfig
from minegs.dataset.cameras import (
    CameraCalibration,
    CameraConvention,
    CameraTable,
    colmap_pose_local_from_cam,
    image_pose_source_from_e57cam,
    pinhole_intrinsics,
)
from minegs.dataset.frames import FrameChain, build_frame_chain, resolve_tls_from_source
from minegs.dataset.staging_input import StagingTree, load_staging
from minegs.eval.protocol import judge
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import crop_equirect
from minegs.ingest.common.geometry import PanoConvention as PanoConv

BUILD_CONFIG_FILE = "build_config.json"
CONVENTION_FILE = "camera_convention.json"
CENTERLINE_FILE = "centerline.csv"
#: Camera centres may sit this far outside the TLS bounding box before §27 complains.
CAMERA_EXTENT_MARGIN_M = 5.0
#: Mapping states that may not enter a dataset (§14). An image in one of these is not dropped;
#: the build stops.
UNRESOLVED_STATUSES = ("unmapped", "ambiguous", "orphan", "conflict")


@dataclass
class StationInfo:
    station_id: str
    scan_id: str
    T_source_from_scanner: SE3
    image_ids: list[str] = field(default_factory=list)

    @property
    def position_source(self) -> np.ndarray:
        return self.T_source_from_scanner.t


@dataclass
class BuildResult:
    dataset_dir: Path
    manifest: Manifest
    report: dict[str, Any]


# ---------------------------------------------------------------------------- selection


def select_stations(tree: StagingTree, representation: str) -> dict[str, StationInfo]:
    """Capture groups from resolved mappings only. Any unresolved image of the wanted
    representation stops the build: silently dropping it would leave a station with fewer
    faces and nobody the wiser (§14)."""
    wanted = [im for im in tree.extraction.image_outputs if im.representation == representation]
    if not wanted:
        have = sorted({im.representation for im in tree.extraction.image_outputs})
        skipped = sorted({s.representation for s in tree.extraction.skipped_images})
        raise ContractError(
            f"{tree.root}: no extracted {representation} images (extracted: {have or 'none'}, "
            f"skipped: {skipped or 'none'}). If the extractor skipped them, re-run "
            "`minegs ingest e57 extract` with a minegs that writes this representation."
        )
    bad = []
    stations: dict[str, StationInfo] = {}
    for im in wanted:
        rec = tree.record(im.image_id)
        if rec.status in UNRESOLVED_STATUSES or not rec.is_resolved:
            bad.append(f"{im.image_id}={rec.status}")
            continue
        scan_id = rec.scan_id
        assert scan_id is not None
        sid = rec.station_id or tree.station_of(scan_id)
        st = stations.get(sid)
        if st is None:
            st = stations[sid] = StationInfo(sid, scan_id, tree.scan_pose(scan_id))
        elif st.scan_id != scan_id:
            raise ContractError(f"station {sid} is claimed by scans {st.scan_id} and {scan_id}")
        st.image_ids.append(im.image_id)
    if bad:
        raise ContractError(
            f"{len(bad)} {representation} image(s) have no resolved station and would be "
            f"dropped silently: {bad[:6]}{'...' if len(bad) > 6 else ''}. Resolve them with an "
            "explicit mapping (minegs ingest e57 pano-map --mapping) or exclude them upstream."
        )
    for st in stations.values():
        st.image_ids.sort()
    return dict(sorted(stations.items()))


# ---------------------------------------------------------------------------- convention


def load_convention(cfg: DatasetBuildConfig) -> CameraConvention:
    cam = cfg.camera
    if cam.convention_file is not None:
        p = Path(cam.convention_file)
        if not p.is_file():
            raise ContractError(f"camera.convention_file {p} not found")
        cal = CameraCalibration.load(p)
        return cal.require_selected(str(p)).with_label()
    assert cam.R_e57cam_from_cam is not None
    try:
        conv = CameraConvention(
            R_e57cam_from_cam=cam.R_e57cam_from_cam, source="explicit", origin="build config"
        )
    except Exception as e:
        raise ContractError(f"camera.R_e57cam_from_cam: {e}") from e
    return conv.with_label()


# ---------------------------------------------------------------------------- centerline


def resolve_centerline(
    cfg: DatasetBuildConfig, chain: FrameChain, tls_sample: np.ndarray
) -> Centerline | None:
    c = cfg.centerline
    if c is None:
        return None
    if c.file is not None:
        p = Path(c.file)
        if not p.is_file():
            raise ContractError(f"centerline.file {p} not found")
        cl = Centerline.from_csv(p, c.frame, "design")
        if c.frame == "SOURCE":
            cl = cl.transformed(chain.T_tls_from_source, "TLS_GLOBAL")
        return cl
    # extracted from the survey itself: straight-ish drifts only, and it says so
    return Centerline.extract_from_points(tls_sample, bin_m=c.bin_m, frame="TLS_GLOBAL")


# ---------------------------------------------------------------------------- split


def resolve_split(cfg: DatasetBuildConfig, order: list[str]) -> tuple[list[str], list[str]]:
    """(train, test) over the ordered station ids. Nothing requested → reconstruction only."""
    s = cfg.split
    if s.test_groups is not None:
        assert s.train_groups is not None
        known = set(order)
        unknown = sorted((set(s.train_groups) | set(s.test_groups)) - known)
        if unknown:
            raise ContractError(f"split names unknown groups {unknown}; groups are {order}")
        unassigned = sorted(known - set(s.train_groups) - set(s.test_groups))
        if unassigned:
            raise ContractError(f"split leaves groups unassigned: {unassigned}")
        return list(s.train_groups), list(s.test_groups)
    if s.test_every is not None:
        n = s.test_every
        test = [g for i, g in enumerate(order) if (i + 1) % n == 0]
        return [g for g in order if g not in test], test
    return list(order), []


# ---------------------------------------------------------------------------- init points


def holdout_mask(s: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(s), dtype=bool)
    for lo, hi in ranges:
        m |= (s >= lo) & (s <= hi)
    return m


def build_init_cloud(
    tree: StagingTree,
    stations: dict[str, StationInfo],
    init_groups: list[str],
    chain: FrameChain,
    centerline_local: Centerline | None,
    holdout: list[tuple[float, float]],
    voxel_m: float | None,
    max_points: int,
    seed: int,
) -> tuple[PointCloud, dict[str, Any]]:
    """LOCAL_METRIC init cloud from the init groups' scans, holdout removed per point."""
    rng = np.random.default_rng(seed)
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    removed = 0
    read = 0
    for gid in init_groups:
        st = stations[gid]
        cloud = read_ply(tree.scan_ply(st.scan_id))
        if cloud.frame != "SOURCE":
            raise ContractError(
                f"{st.scan_id}: scan cloud frame is {cloud.frame!r}, expected SOURCE"
            )
        read += len(cloud)
        xyz = chain.T_local_from_source.apply(cloud.xyz)
        rgb = cloud.rgb if cloud.rgb is not None else np.full((len(cloud), 3), 128, np.uint8)
        if holdout:
            assert centerline_local is not None
            s, _ = centerline_local.project(xyz)
            keep = ~holdout_mask(s, holdout)
            removed += int((~keep).sum())
            xyz, rgb = xyz[keep], rgb[keep]
        if voxel_m:
            idx = voxel_downsample(xyz, voxel_m)
            xyz, rgb = xyz[idx], rgb[idx]
        xyz_parts.append(xyz)
        rgb_parts.append(rgb)
        total = sum(len(p) for p in xyz_parts)
        if total > 4 * max_points:  # bound memory on long surveys; deterministic
            all_xyz = np.concatenate(xyz_parts)
            all_rgb = np.concatenate(rgb_parts)
            keep = np.sort(rng.choice(total, 2 * max_points, replace=False))
            xyz_parts, rgb_parts = [all_xyz[keep]], [all_rgb[keep]]
    if not xyz_parts:
        raise ContractError("no initialization groups; a TLS dataset needs at least one")
    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    if voxel_m and len(xyz_parts) > 1:
        idx = voxel_downsample(xyz, voxel_m)
        xyz, rgb = xyz[idx], rgb[idx]
    if len(xyz) > max_points:
        keep = np.sort(rng.choice(len(xyz), max_points, replace=False))
        xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) == 0:
        raise ContractError("initialization cloud is empty after holdout/voxel filtering")
    stats = {"points_read": read, "points_removed_by_holdout": removed, "points_out": len(xyz)}
    return PointCloud(xyz, rgb, frame="LOCAL_METRIC"), stats


# ---------------------------------------------------------------------------- COLMAP


def build_pinhole_model(
    tree: StagingTree,
    stations: dict[str, StationInfo],
    chain: FrameChain,
    convention: CameraConvention,
    image_ext: dict[str, str],
) -> tuple[colmap_io.ColmapModel, dict[str, list[str]]]:
    table = CameraTable()
    images: dict[int, colmap_io.Image] = {}
    members: dict[str, list[str]] = {}
    img_id = 1
    for sid, st in stations.items():
        names = []
        for image_id in st.image_ids:
            asset = tree.asset(image_id)
            intr = pinhole_intrinsics(asset)
            T_source_from_e57cam = image_pose_source_from_e57cam(asset)
            T_local_from_cam = colmap_pose_local_from_cam(
                chain.T_local_from_source, T_source_from_e57cam, convention
            )
            name = f"{sid}_{image_id}{image_ext[image_id]}"
            images[img_id] = colmap_io.Image.from_world_from_cam(
                img_id, T_local_from_cam, table.id_for(intr), name
            )
            names.append(name)
            img_id += 1
        members[sid] = names
    return colmap_io.ColmapModel(table.cameras, images, {}), members


def spherical_convention(cfg: DatasetBuildConfig) -> PanoConventionConfig:
    """The panorama convention a spherical build uses — the configured one, else the default,
    and either way the one the manifest records."""
    return cfg.camera.pano_convention or PanoConventionConfig()


def build_spherical_model(
    tree: StagingTree,
    stations: dict[str, StationInfo],
    chain: FrameChain,
    cfg: DatasetBuildConfig,
    images_dir: Path,
) -> tuple[colmap_io.ColmapModel, dict[str, list[str]]]:
    """Equirect panorama per station → ring crops (§13). Cylindrical is refused upstream."""
    from PIL import Image as PILImage

    spec = cfg.camera.ring_crop
    assert spec is not None
    pc = spherical_convention(cfg)
    conv = PanoConv(pc.az_sign, pc.el_flip, pc.az_offset_deg, pc.source, pc.vendor)
    cam = colmap_io.Camera.pinhole(1, spec.K(), spec.width, spec.height)
    images: dict[int, colmap_io.Image] = {}
    members: dict[str, list[str]] = {}
    img_id = 1
    for sid, st in stations.items():
        if len(st.image_ids) != 1:
            raise ContractError(
                f"station {sid} has {len(st.image_ids)} spherical images; the ring-crop path "
                "expects exactly one panorama per station"
            )
        asset = tree.asset(st.image_ids[0])
        if asset.representation != "spherical":
            raise ContractError(f"{asset.image_id}: {asset.representation} is not spherical")
        pano = np.asarray(PILImage.open(tree.image_path(asset.image_id)).convert("RGB"))
        T_source_from_scanner = image_pose_source_from_e57cam(asset)
        T_local_from_scanner = chain.T_local_from_source @ T_source_from_scanner
        names = []
        for view in spec.crops():
            name = f"{sid}_{view.name}.png"
            PILImage.fromarray(crop_equirect(pano, view, conv)).save(images_dir / name)
            T_local_from_cam = T_local_from_scanner @ SE3(view.R_scanner_from_cam, np.zeros(3))
            images[img_id] = colmap_io.Image.from_world_from_cam(img_id, T_local_from_cam, 1, name)
            names.append(name)
            img_id += 1
        members[sid] = names
    return colmap_io.ColmapModel({1: cam}, images, {}), members


# ---------------------------------------------------------------------------- sanity (§27)


def sanity_checks(
    model: colmap_io.ColmapModel,
    manifest: Manifest,
    init: PointCloud,
    tls_bounds_local: tuple[np.ndarray, np.ndarray],
    dataset_dir: Path,
) -> dict[str, Any]:
    problems: list[str] = []
    lo, hi = tls_bounds_local
    names = [im.name for im in model.images.values()]
    if len(set(names)) != len(names):
        problems.append("duplicated image names in COLMAP model")
    manifest_images = manifest.all_images()
    if set(names) != set(manifest_images):
        problems.append("COLMAP image set != manifest image set")
    if len(set(manifest_images)) != len(manifest_images):
        problems.append("an image appears in more than one capture group")
    for n in manifest_images:
        if not (dataset_dir / "images" / n).is_file():
            problems.append(f"missing image file {n}")
    centres = []
    for im in model.images.values():
        c = im.center
        if not np.all(np.isfinite(c)):
            problems.append(f"{im.name}: non-finite camera centre")
            continue
        centres.append(c)
        if np.any(c < lo - CAMERA_EXTENT_MARGIN_M) or np.any(c > hi + CAMERA_EXTENT_MARGIN_M):
            problems.append(
                f"{im.name}: camera centre {c.round(2).tolist()} is outside the survey extent"
            )
        try:
            SE3.from_matrix(im.world_from_cam.matrix())
        except Exception as e:  # pragma: no cover - SE3 already validated on construction
            problems.append(f"{im.name}: pose is not rigid ({e})")
    for cam in model.cameras.values():
        K = cam.K()
        if cam.width <= 0 or cam.height <= 0:
            problems.append(f"camera {cam.id}: non-positive dimensions")
        if not (np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0):
            problems.append(f"camera {cam.id}: invalid focal length")
        if not (0 <= K[0, 2] <= cam.width and 0 <= K[1, 2] <= cam.height):
            problems.append(f"camera {cam.id}: principal point outside the image")
    if init.frame != "LOCAL_METRIC":
        problems.append(f"init_points frame is {init.frame}")
    if len(init):
        imin, imax = init.bounds()
        if np.any(imin < lo - 1e-6) or np.any(imax > hi + 1e-6):
            problems.append("init points fall outside the TLS bounds they were cut from")
    try:
        mag = check_float32_safe(
            np.array(centres) if centres else np.zeros((0, 3)), "camera centres"
        )
    except Exception as e:
        problems.append(str(e))
        mag = float("nan")
    if manifest.coordinate_frames.unit != "m":
        problems.append("unit is not metres")
    if problems:
        raise ContractError("dataset sanity checks failed: " + "; ".join(problems))
    return {
        "n_cameras": len(model.cameras),
        "n_images": len(model.images),
        "camera_centre_max_abs_local_m": float(mag),
        "tls_bounds_local": [lo.tolist(), hi.tolist()],
        "init_bounds_local": [b.tolist() for b in init.bounds()] if len(init) else None,
    }


def assert_no_holdout_leak(
    init_ply: Path, centerline_local: Centerline, ranges: list[tuple[float, float]]
) -> int:
    """Re-read the written PLY and prove no point lies in a holdout range (§28)."""
    pc = read_ply(init_ply)
    if pc.frame != "LOCAL_METRIC":
        raise ContractError(f"{init_ply}: frame is {pc.frame}, expected LOCAL_METRIC")
    s, _ = centerline_local.project(pc.xyz)
    leaked = int(holdout_mask(s, ranges).sum())
    if leaked:
        raise ContractError(f"{init_ply}: {leaked} points lie inside holdout ranges {ranges}")
    return len(pc)


# ---------------------------------------------------------------------------- publish (§30)


def _partial_dir(out: Path) -> Path:
    return out.parent / f".{out.name}.minegs-partial"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _publish(tmp: Path, out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise ContractError(f"{out} exists; pass --overwrite to replace it")
        previous = out.parent / f".{out.name}.minegs-previous"
        if previous.exists():
            raise ContractError(f"{previous} exists from an interrupted run; inspect and remove it")
        out.rename(previous)
        try:
            tmp.rename(out)
        except OSError:
            previous.rename(out)
            raise
        shutil.rmtree(previous)
    else:
        tmp.rename(out)


# ---------------------------------------------------------------------------- build


def build_dataset(
    staging: str | Path,
    out: str | Path,
    cfg: DatasetBuildConfig,
    overwrite: bool = False,
    config_path: str | Path | None = None,
) -> BuildResult:
    out = Path(out)
    if out.exists() and not overwrite:
        raise ContractError(f"{out} exists; pass --overwrite to replace it")
    if out.exists() and not out.is_dir():
        raise ContractError(f"{out} is not a directory")
    tmp = _partial_dir(out)
    if tmp.exists():
        raise ContractError(f"{tmp}: a partial build is already there; remove it before building")
    tree = load_staging(staging)
    tmp.mkdir(parents=True)
    try:
        result = _build_into(tree, tmp, out, cfg, config_path)
        _publish(tmp, out, overwrite)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    result.dataset_dir = out
    return result


def _build_into(
    tree: StagingTree, ds: Path, final: Path, cfg: DatasetBuildConfig, config_path
) -> BuildResult:
    representation = "pinhole" if cfg.camera.mode == "e57_pinhole" else "spherical"
    if representation == "spherical":
        cyl = [
            im.image_id
            for im in tree.extraction.image_outputs
            if im.representation == "cylindrical"
        ]
        if cyl:
            raise ContractError(
                f"cylindrical images {cyl[:4]} are present; the cylindrical projection is not "
                "implemented and is not treated as spherical (§13)"
            )
    stations = select_stations(tree, representation)
    convention = load_convention(cfg) if representation == "pinhole" else None

    # ---- frames
    positions_source = np.array([st.position_source for st in stations.values()])
    T_tls_from_source = resolve_tls_from_source(cfg.source_frame)
    positions_tls = T_tls_from_source.apply(positions_source)
    chain = build_frame_chain(cfg.source_frame, cfg.local_metric, positions_tls)

    # ---- TLS extent + centerline (TLS_GLOBAL) + chainage per station
    rng = np.random.default_rng(cfg.initialization.seed)
    tls_lo = np.full(3, np.inf)
    tls_hi = np.full(3, -np.inf)
    sample_parts = []
    for st in stations.values():
        cloud = read_ply(tree.scan_ply(st.scan_id))
        if cloud.frame != "SOURCE":
            raise ContractError(
                f"{st.scan_id}: scan cloud frame is {cloud.frame!r}, expected SOURCE"
            )
        xyz_tls = chain.T_tls_from_source.apply(cloud.xyz)
        tls_lo = np.minimum(tls_lo, xyz_tls.min(axis=0))
        tls_hi = np.maximum(tls_hi, xyz_tls.max(axis=0))
        k = min(len(xyz_tls), 20_000)
        sample_parts.append(xyz_tls[np.sort(rng.choice(len(xyz_tls), k, replace=False))])
    tls_sample = np.concatenate(sample_parts)
    centerline_tls = resolve_centerline(cfg, chain, tls_sample)
    centerline_local = (
        centerline_tls.transformed(chain.T_local_from_tls, "LOCAL_METRIC")
        if centerline_tls
        else None
    )
    chainage: dict[str, float] = {}
    if centerline_tls is not None:
        s, _ = centerline_tls.project(positions_tls)
        chainage = {sid: round(float(v), 3) for sid, v in zip(stations, s, strict=True)}
    holdout = [tuple(r) for r in cfg.geometry_holdout.ranges_m] if cfg.geometry_holdout else []
    if holdout:
        assert centerline_tls is not None
        ext = (centerline_tls.s_start, centerline_tls.s_end)
        outside = [r for r in holdout if r[1] < ext[0] or r[0] > ext[1]]
        if outside:
            raise ContractError(f"holdout ranges {outside} lie outside the centerline extent {ext}")

    # ---- split
    order = sorted(stations, key=(lambda g: (chainage[g], g)) if chainage else (lambda g: g))
    train_groups, test_groups = resolve_split(cfg, order)
    init_groups = list(train_groups)

    # ---- images + COLMAP
    (ds / "images").mkdir(parents=True)
    ext: dict[str, str] = {}
    if representation == "pinhole":
        assert convention is not None
        for st in stations.values():
            for image_id in st.image_ids:
                ext[image_id] = tree.image_path(image_id).suffix.lower()
        model, members = build_pinhole_model(tree, stations, chain, convention, ext)
        for sid, st in stations.items():
            for image_id, name in zip(st.image_ids, members[sid], strict=True):
                _link_or_copy(tree.image_path(image_id), ds / "images" / name, cfg.images.mode)
    else:
        model, members = build_spherical_model(tree, stations, chain, cfg, ds / "images")

    # ---- init points (leak-free) and sparse points from the same set
    init, init_stats = build_init_cloud(
        tree,
        stations,
        init_groups,
        chain,
        centerline_local,
        holdout,
        cfg.initialization.voxel_m,
        cfg.initialization.max_points,
        cfg.initialization.seed,
    )
    write_ply(init, ds / "init_points.ply")
    n_sparse = min(cfg.initialization.sparse_max_points, len(init))
    sp_idx = (
        np.sort(rng.choice(len(init), n_sparse, replace=False))
        if n_sparse < len(init)
        else np.arange(len(init))
    )
    assert init.rgb is not None
    model.points3D = {
        int(i) + 1: colmap_io.Point3D(int(i) + 1, init.xyz[j], init.rgb[j])
        for i, j in enumerate(sp_idx)
    }
    colmap_io.write_model(model, ds / "sparse" / "0")
    if centerline_tls is not None:
        centerline_tls.to_csv(ds / CENTERLINE_FILE)

    # ---- groups / manifest
    groups = {
        sid: CaptureGroup(type="tls_station", members=members[sid], chainage_m=chainage.get(sid))
        for sid in stations
    }
    conv_path = ds / CONVENTION_FILE
    if convention is not None:
        if cfg.camera.convention_file is not None:
            shutil.copy2(cfg.camera.convention_file, conv_path)
        else:
            conv_path.write_text(json.dumps(convention.model_dump(mode="json"), indent=2) + "\n")
    resolved = {
        "config": cfg.model_dump(mode="json"),
        "frames": chain.to_record(),
        "camera_convention": convention.model_dump(mode="json") if convention else None,
        "stations": {
            sid: {"scan_id": st.scan_id, "chainage_m": chainage.get(sid)}
            for sid, st in stations.items()
        },
        "split": {"train_groups": train_groups, "test_groups": test_groups},
        "geometry_holdout_ranges_m": holdout,
        "initialization": init_stats,
        "tls_bounds_tls_global": [tls_lo.tolist(), tls_hi.tolist()],
        "tls_bounds_local_metric": [
            (tls_lo - chain.T_tls_from_local.t).tolist(),
            (tls_hi - chain.T_tls_from_local.t).tolist(),
        ],
        "centerline": None
        if centerline_tls is None
        else {
            "source": centerline_tls.source,
            "frame": "TLS_GLOBAL",
            "chainage_extent_m": [centerline_tls.s_start, centerline_tls.s_end],
        },
        "minegs_version": minegs.__version__,
    }
    cfg_hash = config_hash(resolved["config"])
    (ds / BUILD_CONFIG_FILE).write_text(json.dumps(resolved, indent=2) + "\n")

    assets = _source_assets(
        tree, stations, init_groups, cfg, conv_path if convention else None, ds, config_path
    )
    manifest = Manifest(
        dataset_id=cfg.dataset_id,
        coordinate_frames=CoordinateFrames(T_tls_from_local=chain.T_tls_from_local.to_list()),
        capture_groups=groups,
        split=Split(
            train_groups=train_groups,
            test_groups=test_groups,
            geometry_holdout=GeometryHoldout(
                chainage_ranges_m=holdout,
                points_excluded=True,
                images_excluded=cfg.geometry_holdout.images_excluded,
            )
            if holdout
            else None,
        ),
        initialization=Initialization(
            source="tls",
            file="init_points.ply",
            groups=init_groups,
            excluded_chainage_ranges_m=holdout,
            n_points=len(init),
        ),
        provenance=ManifestProvenance(
            minegs_version=minegs.__version__,
            git_commit=git_commit(),
            config_hash=cfg_hash,
            source_assets=assets,
            tool_versions=tool_versions(),
        ),
        source="tls",
        capture_epoch=CaptureEpoch(id=cfg.capture_epoch.id, date=cfg.capture_epoch.date)
        if cfg.capture_epoch
        else None,
        scale=Scale(basis="tls_pose", factor=1.0),
        centerline=CenterlineRef(
            file=CENTERLINE_FILE, source=centerline_tls.source, frame="TLS_GLOBAL"
        )
        if centerline_tls
        else None,
    )
    if representation == "spherical":
        pc = spherical_convention(cfg)
        manifest.pano_convention = PanoConvention(
            az_sign=pc.az_sign,
            el_flip=pc.el_flip,
            az_offset=pc.az_offset_deg,
            source=pc.source,
            vendor=pc.vendor,
        )
    manifest.save_dataset(ds)

    # ---- verification before publication (§27–§29)
    loaded = Manifest.load_dataset(ds)
    issues = loaded.consistency_issues()
    if issues:
        raise ContractError("manifest consistency: " + "; ".join(issues))
    judgement = judge(loaded)
    lo_local = tls_lo - chain.T_tls_from_local.t
    hi_local = tls_hi - chain.T_tls_from_local.t
    checks = sanity_checks(
        colmap_io.read_model(ds / "sparse" / "0"), loaded, init, (lo_local, hi_local), ds
    )
    leak_checked = None
    if holdout:
        assert centerline_local is not None
        leak_checked = assert_no_holdout_leak(ds / "init_points.ply", centerline_local, holdout)
        # and the sparse points, which are cut from the same set
        s, _ = centerline_local.project(np.array([p.xyz for p in model.points3D.values()]))
        if holdout_mask(s, holdout).any():
            raise ContractError("points3D.txt would reintroduce holdout geometry")
    leak = sorted(set(loaded.initialization.groups) & set(loaded.split.test_groups))
    if leak:
        raise ContractError(f"initialization uses test groups {leak}")
    report = {
        "dataset_id": cfg.dataset_id,
        "dataset_dir": str(final),
        "config_hash": cfg_hash,
        "frames": chain.to_record(),
        "camera_convention": convention.model_dump(mode="json") if convention else None,
        "n_stations": len(stations),
        "n_images": len(model.images),
        "split": {"train_groups": train_groups, "test_groups": test_groups},
        "geometry_holdout_ranges_m": holdout,
        "initialization": init_stats,
        "init_points_holdout_checked": leak_checked,
        "protocols": [p.value for p in judgement.protocols],
        "claims": [c.value for c in judgement.claims],
        "refusals": judgement.refusals,
        "sanity": checks,
    }
    return BuildResult(final, loaded, report)


def _source_assets(
    tree, stations, init_groups, cfg, conv_path, ds, config_path
) -> list[SourceAsset]:
    assets: list[SourceAsset] = [
        SourceAsset(path=tree.extraction.source_e57, sha256=tree.source_sha256),
        *tree.artifact_assets,
    ]
    for mi in tree.mapping.mapping_inputs:
        if mi.sha256 is None:
            raise ContractError(f"mapping input {mi.path} has no digest ({mi.hash_skipped_reason})")
        assets.append(SourceAsset(path=mi.path, sha256=mi.sha256, size_bytes=mi.size_bytes))
    for gid in sorted(set(stations)):
        st = stations[gid]
        so = tree.scan_output(st.scan_id)
        assert so.sha256 is not None
        assets.append(SourceAsset(path=so.path, sha256=so.sha256))
        for image_id in st.image_ids:
            io = tree.image_output(image_id)
            if io.sha256 is None:
                raise ContractError(
                    f"image {image_id} has no digest ({io.hash_skipped_reason}); provenance cannot name it"
                )
            assets.append(SourceAsset(path=io.path, sha256=io.sha256, size_bytes=io.bytes_written))
    if conv_path is not None:
        assets.append(
            SourceAsset(
                path=str(conv_path.name),
                sha256=sha256_file(conv_path),
                size_bytes=conv_path.stat().st_size,
            )
        )
    if cfg.centerline is not None and cfg.centerline.file is not None:
        p = Path(cfg.centerline.file)
        assets.append(SourceAsset(path=str(p), sha256=sha256_file(p), size_bytes=p.stat().st_size))
    if config_path is not None:
        p = Path(config_path)
        assets.append(SourceAsset(path=str(p), sha256=sha256_file(p), size_bytes=p.stat().st_size))
    return assets
