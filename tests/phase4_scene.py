"""Phase 4 test scenes: a TLS dataset, and an image-only dataset whose SfM has tracks.

The Phase 3 stand-in SfM writes points with no observation tracks, which is all registration
and initialisation need. Depth supervision from SfM needs the tracks, so this wraps it: every
point is observed by the frames it projects into, near enough to have been matched, with the
keypoint a fraction of a pixel off the reprojection, as a real feature detector would leave it.

None of this is G2 or G3. The stand-ins draw the answer they were given.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from minegs.core.pointcloud import PointCloud, write_ply
from minegs.core.synthetic import SyntheticSpec, generate
from minegs.core.synthetic_staging import StagingSpec, generate_staging
from minegs.ingest.common import colmap_io

from video_survey import T_SFM_FROM_TLS, generate_video_survey, stand_in_sfm

THRESHOLDS = {"max_rmse_m": 0.10, "min_inlier_ratio": 0.6, "min_correspondences": 6}


def tls_scene(root: Path) -> SimpleNamespace:
    """The Phase 0A synthetic tunnel: a TLS dataset with a holdout and a centerline."""
    r = generate(
        root, SyntheticSpec(length_m=90.0, station_spacing_m=15.0, image_size=48, points_per_m=1500)
    )
    return SimpleNamespace(
        dataset_dir=r.dataset_dir, cloud=r.root / "raw" / "tls_full.ply", result=r
    )


def stand_in_sfm_with_tracks(
    survey, *, max_points: int = 4000, reach_m: float = 14.0, noise_px: float = 0.2, seed: int = 11
):
    base = stand_in_sfm(survey, max_points=max_points, seed=seed)

    def _run(images_dir: Path, work_dir: Path, opts):
        run = base(images_dir, work_dir, opts)
        m = colmap_io.read_model(work_dir / "sparse/0")
        rng = np.random.default_rng(seed)
        ids = sorted(m.points3D)
        xyz = np.array([m.points3D[i].xyz for i in ids])
        tracks: dict[int, list[tuple[int, int]]] = {i: [] for i in ids}
        for im in m.images.values():
            cam = m.cameras[im.camera_id]
            uv, z = colmap_io.project(cam.K(), im.cam_from_world, xyz)
            vis = colmap_io.visible_mask(uv, z, cam.width, cam.height, near=1e-6)
            vis &= z < reach_m * T_SFM_FROM_TLS.s
            js = np.flatnonzero(vis)
            im.xys = uv[js] + rng.normal(0, noise_px, (len(js), 2))
            im.point3D_ids = np.array([ids[j] for j in js], dtype=np.int64)
            for k, j in enumerate(js):
                tracks[ids[j]].append((im.id, k))
        for pid, t in tracks.items():
            p = m.points3D[pid]
            p.image_ids = np.array([a for a, _ in t], dtype=np.int64)
            p.point2D_idxs = np.array([b for _, b in t], dtype=np.int64)
        colmap_io.write_model(m, work_dir / "sparse/0")
        return run

    return _run


def video_scene(root: Path, *, holdout=(20.0, 26.0), images_excluded=True, group_size=3):
    """A plain-video image-only dataset over the Phase 2 scene, built through the library."""
    from minegs.dataset.from_sfm import SfmDatasetConfig, build_dataset_from_sfm
    from minegs.eval.register.run import register_sfm
    from minegs.ingest.video.build import build_frameset
    from minegs.ingest.video.sfm.run import run_sfm

    root = Path(root)
    staging = generate_staging(
        root / "synthetic",
        StagingSpec(length_m=60.0, station_spacing_m=12.0, points_per_m=3000, image_size=64),
    )
    centerline_csv = root / "centerline_source.csv"
    staging.centerline_source.to_csv(centerline_csv)
    tls = write_ply(
        PointCloud(staging.points_source.xyz, rgb=staging.points_source.rgb, frame="TLS_GLOBAL"),
        root / "raw" / "tls_full.ply",
        xyz_dtype="f8",
    )
    survey = generate_video_survey(
        root / "video", staging.centerline_source, staging.points_source.xyz
    )
    _, fs_dir = build_frameset(
        root / "fs",
        kind="image_set",
        image_dir=survey.frames_dir,
        blur_threshold=0.0,
        hamming_threshold=0,
    )
    _, sfm_dir = run_sfm(fs_dir, root / "sfm", executor=stand_in_sfm_with_tracks(survey))
    _, reg_dir = register_sfm(
        sfm_dir,
        root / "reg",
        basis="known_target",
        targets_sfm=survey.targets_sfm,
        targets_tls=survey.targets_tls,
        reference_ply=tls,
        thresholds=dict(THRESHOLDS),
    )
    cfg = SfmDatasetConfig(
        dataset_id="p4_video",
        source="video",
        group_size=group_size,
        geometry_holdout_m=[holdout],
        holdout_images_excluded=images_excluded,
        centerline_file=str(centerline_csv),
        centerline_source="design",
        init_voxel_m=0.05,
    )
    manifest, ds = build_dataset_from_sfm(fs_dir, sfm_dir, reg_dir, root / "ds", cfg)
    return SimpleNamespace(
        dataset_dir=Path(ds), manifest=manifest, tls=tls, survey=survey, sfm_dir=sfm_dir
    )
