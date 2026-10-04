"""Builders for ``DepthSupervisionRecord`` (Phase 4 C1, docs/PHASE4_CONTRACT.md §5).

Two sources, each from evidence the dataset already holds or that is named explicitly:

* ``sfm_tracks``: the image-only path's own reconstruction. The bundled ``SFM_INTERNAL`` model
  has an arbitrary scale, so it is put into metres by the measured registration before any
  depth is read from it. Each sample is a track point reprojected through the dataset camera
  that observed it, which is how upstream forms its targets too.
* ``tls_projection``: a materialised TLS cloud projected into the TLS dataset's own training
  cameras, with an explicit visibility rule.

Every builder ends by running the verifier on what it wrote. A builder therefore cannot emit
an artifact that training would refuse. The leakage rules are the same code on both sides
(``support.Support``).
"""

from __future__ import annotations

import os
import shutil
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import minegs
from minegs.core.errors import ContractError
from minegs.core.frames import Sim3
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import git_commit, make_id, sha256_file
from minegs.ingest.common import colmap_io
from minegs.train.supervision.depth import (
    BINARY,
    DEPTH_SEMANTICS,
    DEPTH_UNIT,
    FRAME,
    IMAGE_ONLY_SOURCES,
    PINHOLE_MODELS,
    PIXEL_CONVENTION,
    RECORD_FILE,
    SAMPLE_DTYPE,
    SAMPLES_FILE,
    SUPERVISION_DIR,
    DepthSupervisionRecord,
    SourceAssetRef,
    VerifiedDepthSupervision,
    dataset_binding,
    recorded_support,
    training_image_names,
    verify_depth_supervision,
)
from minegs.train.supervision.support import (
    DEFAULT_MAX_RADIAL_M,
    DEFAULT_RAY_MARGIN_M,
    HOLDOUT_POINT,
    HOLDOUT_RAY,
    UNLOCATED,
    Support,
)

# Quality reasons: the sample is kept for audit, with confidence 0 (AD-4).
SHORT_TRACK = "short_track"
REPROJECTION = "reprojection_error"
NO_KEYPOINT = "no_keypoint"
# Exclusion reasons that are not leakage: the sample is not emitted.
OUT_OF_FRAME = "out_of_frame"
NON_TRAINING_TRACK = "track_touches_non_training_image"
OCCLUDED = "occluded"
BUDGET = "over_per_image_budget"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _train_cameras(manifest: Manifest, model: colmap_io.ColmapModel) -> list[str]:
    """Training images present in the model, in a fixed order, all pinhole."""
    by_name = model.image_by_name()
    names = training_image_names(manifest, model)
    for n in names:
        cam = model.cameras[by_name[n].camera_id]
        if cam.model not in PINHOLE_MODELS:
            raise ContractError(
                f"{n} uses camera model {cam.model}; depth supervision is defined for "
                f"{PINHOLE_MODELS} only (the renderer's restriction)"
            )
    if not names:
        raise ContractError("the dataset has no training images to supervise")
    return names


def _budget(name: str, n: int, budget: int | None, seed: int) -> np.ndarray:
    """Deterministic subsample of n rows to at most `budget`, seeded by the image name."""
    if budget is None or n <= budget:
        return np.arange(n)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    return np.sort(rng.choice(n, budget, replace=False))


def _write(
    dataset_dir: Path,
    out_dir: Path | None,
    manifest: Manifest,
    model: colmap_io.ColmapModel,
    support: Support,
    *,
    source_kind: str,
    source_assets: list[SourceAssetRef],
    images: list[str],
    rows: list[np.ndarray],
    exclusions: dict[str, int],
    params: dict[str, Any],
) -> VerifiedDepthSupervision:
    """Assemble, write atomically, then verify what is on disk."""
    samples = np.concatenate(rows) if rows else np.zeros(0, dtype=SAMPLE_DTYPE)
    # Images that ended up with no row are dropped from the index space, so `images` is
    # exactly the set of images the artifact speaks about.
    used = sorted(set(samples["image"].astype(np.int64).tolist()))
    remap = np.full(len(images), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    samples = samples.copy()
    samples["image"] = remap[samples["image"].astype(np.int64)].astype(np.uint32)
    images = [images[i] for i in used]
    if not len(samples):
        raise ContractError(
            f"no depth sample survived (exclusions: {exclusions}); an empty supervision "
            "artifact would make a depth-supervised run indistinguishable from one without"
        )
    order = np.lexsort((samples["v"], samples["u"], samples["image"]))
    samples = samples[order]

    sid = make_id("dsup")
    out = Path(out_dir) if out_dir is not None else dataset_dir / SUPERVISION_DIR / sid
    if out.exists() and any(out.iterdir()):
        raise ContractError(f"{out} is not empty; an artifact is written once")
    partial = out.parent / f".{out.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.save(partial / SAMPLES_FILE, samples, allow_pickle=False)
    counts = np.bincount(samples["image"].astype(np.int64), minlength=len(images)).tolist()
    record = DepthSupervisionRecord(
        supervision_id=sid,
        dataset_id=manifest.dataset_id,
        dataset_binding=dataset_binding(dataset_dir, manifest, model),
        source_kind=source_kind,
        source_assets=source_assets,
        frame=FRAME,
        depth_unit=DEPTH_UNIT,
        depth_semantics=DEPTH_SEMANTICS,
        pixel_convention=PIXEL_CONVENTION,
        confidence_semantics=BINARY,
        images=images,
        n_samples=len(samples),
        n_samples_in_loss=int((samples["confidence"] > 0).sum()),
        per_image_counts=counts,
        support_ranges_m=recorded_support(samples, images, model, support),
        excluded_holdout_ranges_m=list(support.holdout),
        exclusions={k: int(v) for k, v in sorted(exclusions.items())},
        creation_params=params,
        minegs_version=minegs.__version__,
        git_commit=git_commit(),
        created_at=_now(),
        samples_file=SAMPLES_FILE,
        samples_sha256=sha256_file(partial / SAMPLES_FILE),
    )
    record.save(partial / RECORD_FILE)
    if out.exists():
        out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, out)
    return verify_depth_supervision(dataset_dir, out, manifest=manifest, model=model)


def _rows(image_index: int, uv: np.ndarray, z: np.ndarray, conf: np.ndarray) -> np.ndarray:
    r = np.zeros(len(z), dtype=SAMPLE_DTYPE)
    r["image"] = image_index
    r["u"] = uv[:, 0]
    r["v"] = uv[:, 1]
    r["depth_m"] = z
    r["confidence"] = conf
    return r


def _in_frame(uv: np.ndarray, z: np.ndarray, cam: colmap_io.Camera, min_depth: float) -> np.ndarray:
    # Strictly inside, in float32 as stored: a coordinate that rounds onto the far edge would
    # be refused by the verifier, so it is not emitted.
    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)
    return (z > min_depth) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)


# ================================================================ sfm_tracks


def build_sfm_tracks(
    dataset_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    min_track_length: int = 3,
    max_reproj_error_px: float = 2.0,
    min_depth_m: float = 0.05,
    max_samples_per_image: int | None = None,
    max_radial_m: float = DEFAULT_MAX_RADIAL_M,
    ray_margin_m: float = DEFAULT_RAY_MARGIN_M,
    seed: int = 0,
) -> VerifiedDepthSupervision:
    """Depth from the image-only dataset's own SfM tracks, made metric by its registration."""
    from minegs.dataset.from_sfm import PROVENANCE_DIR, SFM_MODEL_DIR, check_image_only_dataset
    from minegs.eval.register.models import RegistrationRecord

    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds)
    if manifest.source not in IMAGE_ONLY_SOURCES:
        raise ContractError(
            f"{ds} has source {manifest.source!r}; sfm_tracks depth comes from an image-only "
            "dataset's own reconstruction, and a TLS dataset has none"
        )
    check_image_only_dataset(ds)
    prov = ds / PROVENANCE_DIR
    sfm_dir = prov / SFM_MODEL_DIR
    reg_json = prov / "registration.json"
    rec_reg = RegistrationRecord.load(reg_json)
    # SFM_INTERNAL has an arbitrary scale. The measured Sim(3) is the only way into metres,
    # and it is the one the dataset itself was built with.
    T_local_from_sfm = Sim3.from_se3(manifest.T_local_from_tls) @ rec_reg.sim3()

    model = colmap_io.read_model(ds / "sparse" / "0")
    sfm = colmap_io.read_model(sfm_dir)
    support = Support.of_dataset(ds, manifest, max_radial_m=max_radial_m, ray_margin_m=ray_margin_m)
    names = _train_cameras(manifest, model)
    index = {n: i for i, n in enumerate(names)}
    train = set(names)
    by_name = model.image_by_name()
    sfm_name = {im.id: im.name for im in sfm.images.values()}

    exclusions: dict[str, int] = {}

    def count(reason: str, n: int = 1) -> None:
        if n:
            exclusions[reason] = exclusions.get(reason, 0) + int(n)

    per_image: dict[int, list[tuple[float, float, float, float]]] = {}
    for p in sfm.points3D.values():
        track = [sfm_name.get(int(i)) for i in p.image_ids]
        if not track:
            continue
        # A depth triangulated with a held-out view carries that view's information.
        if any(n is None or n not in train for n in track):
            count(NON_TRAINING_TRACK)
            continue
        X = T_local_from_sfm.apply(np.asarray(p.xyz, dtype=np.float64).reshape(1, 3))[0]
        short = len(set(track)) < min_track_length
        for iid, p2d, name in zip(p.image_ids, p.point2D_idxs, track, strict=True):
            im = by_name[name]
            cam = model.cameras[im.camera_id]
            uv, z = colmap_io.project(cam.K(), im.cam_from_world, X.reshape(1, 3))
            if not _in_frame(uv, z, cam, min_depth_m)[0]:
                count(OUT_OF_FRAME)
                continue
            conf = 1.0
            sim = sfm.images[int(iid)]
            if short:
                conf = 0.0
                count(SHORT_TRACK)
            elif not (0 <= int(p2d) < len(sim.xys)):
                conf = 0.0
                count(NO_KEYPOINT)
            elif float(np.linalg.norm(uv[0] - sim.xys[int(p2d)])) > max_reproj_error_px:
                conf = 0.0
                count(REPROJECTION)
            per_image.setdefault(index[name], []).append(
                (float(uv[0, 0]), float(uv[0, 1]), float(z[0]), conf)
            )

    rows: list[np.ndarray] = []
    for k in sorted(per_image):
        arr = np.array(per_image[k], dtype=np.float64)
        rows.append(_rows(k, arr[:, :2], arr[:, 2], arr[:, 3]))
    rows = _apply_support(rows, names, model, support, count)
    rows = [
        r[_budget(names[int(r["image"][0])], len(r), max_samples_per_image, seed)]
        for r in rows
        if len(r)
    ]
    sfm_files = sorted(p for p in sfm_dir.iterdir() if p.is_file())
    assets = [
        SourceAssetRef(
            role="sfm_model",
            path=f"{PROVENANCE_DIR}/{SFM_MODEL_DIR}/{f.name}",
            sha256=sha256_file(f),
        )
        for f in sfm_files
    ] + [
        SourceAssetRef(
            role="registration",
            path=f"{PROVENANCE_DIR}/registration.json",
            sha256=sha256_file(reg_json),
        )
    ]
    params = {
        "min_track_length": int(min_track_length),
        "max_reproj_error_px": float(max_reproj_error_px),
        "min_depth_m": float(min_depth_m),
        "max_samples_per_image": max_samples_per_image,
        "max_radial_m": float(max_radial_m),
        "ray_margin_m": float(ray_margin_m),
        "seed": int(seed),
        "pixel": "reprojection of the metric track point through the dataset camera",
        "registration_id": rec_reg.registration_id,
        "registration_scale": float(rec_reg.sim3().s),
    }
    return _write(
        ds,
        Path(out_dir) if out_dir else None,
        manifest,
        model,
        support,
        source_kind="sfm_tracks",
        source_assets=assets,
        images=names,
        rows=rows,
        exclusions=exclusions,
        params=params,
    )


def _apply_support(
    rows: list[np.ndarray],
    names: list[str],
    model: colmap_io.ColmapModel,
    support: Support,
    count,
) -> list[np.ndarray]:
    """Drop every sample the leakage rules refuse, judged on the float32 values as stored."""
    from minegs.train.supervision.depth import backproject_samples

    out = []
    for r in rows:
        if not len(r):
            continue
        pts, cams = backproject_samples(r, names, model)
        keep, _, reasons = support.classify(cams, pts)
        for reason in (UNLOCATED, HOLDOUT_POINT, HOLDOUT_RAY):
            count(reason, int(reasons[reason].sum()))
        out.append(r[keep])
    return out


# ================================================================ tls_projection


def build_tls_projection(
    dataset_dir: str | Path,
    cloud_ply: str | Path,
    out_dir: str | Path | None = None,
    *,
    cell_px: int = 4,
    occlusion_kernel_cells: int = 3,
    occlusion_rel_tol: float = 0.15,
    min_depth_m: float = 0.1,
    max_depth_m: float | None = None,
    max_samples_per_image: int | None = 8192,
    max_radial_m: float = DEFAULT_MAX_RADIAL_M,
    ray_margin_m: float = DEFAULT_RAY_MARGIN_M,
    seed: int = 0,
) -> VerifiedDepthSupervision:
    """Depth from a named TLS cloud, projected into the TLS dataset's training cameras.

    Visibility: a ``cell_px`` grid keeps the nearest point of each cell; a cell whose nearest
    point is more than ``occlusion_rel_tol`` further than the nearest in its
    ``occlusion_kernel_cells`` neighbourhood is treated as seen *through* a gap between sparse
    near-surface points, and dropped. Holdout and unlocated points are removed before
    projection; rays through the holdout are removed after (AD-3).
    """
    from scipy.ndimage import minimum_filter

    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds)
    if manifest.source in IMAGE_ONLY_SOURCES:
        raise ContractError(
            f"{ds} is a {manifest.source} dataset. TLS depth would make the image-only path "
            "TLS-assisted (Phase 3 independence); use sfm_tracks for it."
        )
    if cell_px < 1 or occlusion_kernel_cells < 1 or occlusion_kernel_cells % 2 == 0:
        raise ContractError("cell_px >= 1 and an odd occlusion_kernel_cells >= 1 are required")
    cloud_ply = Path(cloud_ply).resolve()
    cloud_sha = sha256_file(cloud_ply)
    init_path = ds / manifest.initialization.file
    if init_path.is_file() and sha256_file(init_path) == cloud_sha:
        raise ContractError(
            f"{cloud_ply} is the dataset's initialisation {manifest.initialization.file}. "
            "Depth projected from the initial geometry would be the init standing in for "
            "evidence (contract AD-1); project the TLS cloud the init was sampled from."
        )
    cloud = read_ply(cloud_ply)
    if cloud.frame == "LOCAL_METRIC":
        xyz = np.asarray(cloud.xyz, dtype=np.float64)
    elif cloud.frame == "TLS_GLOBAL":
        xyz = manifest.T_local_from_tls.apply(np.asarray(cloud.xyz, dtype=np.float64))
    else:
        raise ContractError(
            f"{cloud_ply} declares frame {cloud.frame!r}; a TLS cloud for depth must say whether "
            "it is LOCAL_METRIC or TLS_GLOBAL, because depth in an unknown frame is not metres"
        )
    model = colmap_io.read_model(ds / "sparse" / "0")
    support = Support.of_dataset(ds, manifest, max_radial_m=max_radial_m, ray_margin_m=ray_margin_m)
    exclusions: dict[str, int] = {}

    def count(reason: str, n: int = 1) -> None:
        if n:
            exclusions[reason] = exclusions.get(reason, 0) + int(n)

    # Holdout geometry never enters the projection (AD-3): the ray rule below guarantees that
    # no remaining sample looks through the space it occupied.
    if support.required:
        s, ok = support.locate(xyz)
        count("cloud_" + UNLOCATED, int((~ok).sum()))
        inside = ok & support.in_holdout(s)
        count("cloud_" + HOLDOUT_POINT, int(inside.sum()))
        xyz = xyz[ok & ~inside]

    names = _train_cameras(manifest, model)
    by_name = model.image_by_name()
    rows: list[np.ndarray] = []
    for k, name in enumerate(names):
        im = by_name[name]
        cam = model.cameras[im.camera_id]
        uv, z = colmap_io.project(cam.K(), im.cam_from_world, xyz)
        vis = _in_frame(uv, z, cam, min_depth_m)
        if max_depth_m is not None:
            vis &= z <= max_depth_m
        if not vis.any():
            continue
        uv, z = uv[vis], z[vis]
        gw = int(np.ceil(cam.width / cell_px))
        gh = int(np.ceil(cam.height / cell_px))
        cx = np.clip((uv[:, 0] // cell_px).astype(np.int64), 0, gw - 1)
        cy = np.clip((uv[:, 1] // cell_px).astype(np.int64), 0, gh - 1)
        cell = cy * gw + cx
        order = np.lexsort((z, cell))  # by cell, nearest first
        first = np.ones(len(order), dtype=bool)
        first[1:] = cell[order][1:] != cell[order][:-1]
        near = order[first]
        zmin = np.full(gh * gw, np.inf)
        zmin[cell[near]] = z[near]
        neigh = minimum_filter(
            zmin.reshape(gh, gw), size=occlusion_kernel_cells, mode="constant", cval=np.inf
        ).reshape(-1)
        seen = z[near] <= neigh[cell[near]] * (1.0 + occlusion_rel_tol)
        count(OCCLUDED, int((~seen).sum()))
        near = near[seen]
        r = _rows(k, uv[near], z[near], np.ones(len(near)))
        r = r[np.lexsort((r["v"], r["u"]))]
        rows.append(r)
    rows = _apply_support(rows, names, model, support, count)
    kept = []
    for r in rows:
        if not len(r):
            continue
        sel = _budget(names[int(r["image"][0])], len(r), max_samples_per_image, seed)
        count(BUDGET, len(r) - len(sel))
        kept.append(r[sel])
    assets = [SourceAssetRef(role="tls_cloud", path=str(cloud_ply), sha256=cloud_sha)]
    params = {
        "cloud_frame": cloud.frame,
        "cell_px": int(cell_px),
        "occlusion_kernel_cells": int(occlusion_kernel_cells),
        "occlusion_rel_tol": float(occlusion_rel_tol),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": max_depth_m,
        "max_samples_per_image": max_samples_per_image,
        "max_radial_m": float(max_radial_m),
        "ray_margin_m": float(ray_margin_m),
        "seed": int(seed),
        "pixel": "projection of the nearest visible TLS point per cell, at its own position",
    }
    return _write(
        ds,
        Path(out_dir) if out_dir else None,
        manifest,
        model,
        support,
        source_kind="tls_projection",
        source_assets=assets,
        images=names,
        rows=kept,
        exclusions=exclusions,
        params=params,
    )


__all__ = ["build_sfm_tracks", "build_tls_projection"]
