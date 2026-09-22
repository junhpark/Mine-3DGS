"""A synthetic 360 survey of the Phase 2 scene, and the stand-ins the Phase 3 gate needs.

The point of building the image survey *from the same scene* as the scanner survey is that the
two paths can then be compared: one tunnel, one reference axis, one held-out TLS cloud, two
reconstructions of it. A drift photographed here and scanned somewhere else would give two
numbers nobody could subtract.

Three things are stood in for, each because this machine cannot do the real thing, and each
recorded as a substitution by the code that consumes it:

* **frame extraction** — ``copy_extractor`` takes the panoramas from a directory instead of
  decoding a video, and ``ExtractionRecord.real_execution`` comes out false;
* **SfM** — :func:`stand_in_sfm` writes the model a successful reconstruction would have
  produced: the true camera poses and wall points, put through one arbitrary similarity, so
  the result is in ``SFM_INTERNAL`` at a scale nothing has measured yet. Registration has to
  recover that similarity from survey control alone, which is the thing being tested;
* **the depth renderer** — :class:`TunnelDepthRenderer` z-buffers the reference cloud through
  each camera. It presents as the renderer this build ships, because promotion requires that,
  and ``render_depths`` records ``real_renderer_execution: false`` regardless.

**None of this is G2.** A stand-in that draws the answer it was given is evidence about the
pipeline's wiring and about nothing else.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from minegs.core.centerline import Centerline
from minegs.core.frames import SE3, Sim3, rot_z, rotmat_to_quat
from minegs.core.synthetic_staging import render_equirect, wall_colour
from minegs.eval.surface.render import DepthRenderer, GsplatDepthRenderer
from minegs.ingest.common import colmap_io
from minegs.ingest.common.equirect import RingCropSpec
from minegs.ingest.common.geometry import PanoConvention

#: The panorama convention this camera is declared to use. Named as a camera's own — the
#: library default claims ``E57Embedded``, and ``build_frameset`` refuses that for a video set.
CAMERA_CONVENTION = PanoConvention(
    az_sign=1,
    el_flip=False,
    az_offset_deg=0.0,
    source="Configured",
    vendor="synthetic_360_rig",
)

RING = RingCropSpec(n_yaw=4, fov_deg=90.0, width=64, height=64)

#: The arbitrary similarity a reconstruction comes out in. Scale well away from 1 so a pipeline
#: that quietly assumes metric produces something obviously wrong rather than something close.
T_SFM_FROM_TLS = Sim3(0.37, rot_z(23.0), np.array([12.0, -5.0, 3.0]))


@dataclass
class VideoSurvey:
    root: Path
    video: Path
    frames_dir: Path
    centerline: Centerline
    pano_poses: dict[str, SE3]  # frame file stem -> T_tls_from_scanner
    points_tls: np.ndarray
    control_tls: np.ndarray
    targets_sfm: Path
    targets_tls: Path
    #: Where the survey control sits along the axis. Declared, because a support extent nobody
    #: recorded makes overlap with the evaluation holdout undecidable (§Phase 3 AD-2).
    control_ranges_m: list[tuple[float, float]]
    ring: RingCropSpec
    convention: PanoConvention


def _write_targets(path: Path, ids: list[str], pts: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z"])
        for i, p in zip(ids, pts, strict=True):
            w.writerow([i, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
    return path


def generate_video_survey(
    root: str | Path,
    centerline: Centerline,
    points_tls: np.ndarray,
    *,
    spacing_m: float = 5.0,
    pano_width: int = 256,
    pano_height: int = 128,
    render_points: int = 60_000,
    n_control: int = 8,
    control_noise_m: float = 0.004,
    seed: int = 7,
) -> VideoSurvey:
    """Walk the drift, render a panorama every *spacing_m*, and place the survey control.

    The panoramas are written where a decoder would have written them; the video file beside
    them exists so the capture has a digest, and the frame set records that no decoder ran.
    """
    root = Path(root)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    xyz = np.asarray(points_tls, dtype=np.float64)
    if len(xyz) > render_points:
        xyz = xyz[np.sort(rng.choice(len(xyz), render_points, replace=False))]
    rgb = wall_colour(xyz)

    stations = np.arange(centerline.s_start + spacing_m / 2, centerline.s_end, spacing_m)
    poses: dict[str, SE3] = {}
    for i, s in enumerate(stations):
        T = centerline.frame_at(float(s))  # x tangent, y left, z up — the scanner frame
        stem = f"v_{i + 1:06d}"
        local = T.inverse().apply(xyz)
        img = render_equirect(local, rgb, pano_width, pano_height, CAMERA_CONVENTION)
        from PIL import Image

        Image.fromarray(img).save(frames_dir / f"{stem}.png")
        poses[stem] = T

    video = root / "drift_360.mp4"
    video.write_bytes(b"a 360 capture of this drift; its frames were extracted elsewhere\n")

    # Survey control: prisms on the wall at known chainage, not points read off the reference
    # cloud. Their extent is what the registration support covers.
    lo, hi = centerline.s_start + 2.0, centerline.s_start + 10.0
    cs = np.linspace(lo, hi, n_control)
    control = []
    for k, s in enumerate(cs):
        T = centerline.frame_at(float(s))
        r, th = 2.4, 2 * np.pi * k / n_control
        control.append(T.apply(np.array([[0.0, r * np.cos(th), r * np.sin(th)]]))[0])
    control_tls = np.array(control)
    ids = [f"cp{k:02d}" for k in range(len(control_tls))]
    control_sfm = T_SFM_FROM_TLS.apply(control_tls)
    control_sfm = control_sfm + rng.normal(0, control_noise_m * T_SFM_FROM_TLS.s, control_sfm.shape)
    return VideoSurvey(
        root=root,
        video=video,
        frames_dir=frames_dir,
        centerline=centerline,
        pano_poses=poses,
        points_tls=np.asarray(points_tls, dtype=np.float64),
        control_tls=control_tls,
        targets_sfm=_write_targets(root / "targets_sfm.csv", ids, control_sfm),
        targets_tls=_write_targets(root / "targets_tls.csv", ids, control_tls),
        control_ranges_m=[(float(lo), float(hi))],
        ring=RING,
        convention=CAMERA_CONVENTION,
    )


def stand_in_sfm(survey: VideoSurvey, *, max_points: int = 6000, seed: int = 11):
    """The model a successful reconstruction would have left, in a frame nothing has measured.

    Poses and points are the truth put through one similarity, which is exactly the freedom a
    real SfM has: correct up to scale, rotation and translation. Registration is then a real
    measurement of a real unknown, rather than a check that the identity is the identity.
    """
    from minegs.ingest.video.sfm.base import SfMRun

    def _run(images_dir: Path, work_dir: Path, _opts) -> SfMRun:
        rng = np.random.default_rng(seed)
        views = {v.name: v for v in survey.ring.crops()}
        K = survey.ring.K()
        cameras = {
            i + 1: colmap_io.Camera(
                i + 1,
                "PINHOLE",
                survey.ring.width,
                survey.ring.height,
                [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
            )
            for i in range(len(views))
        }
        camera_id = {name: i + 1 for i, name in enumerate(sorted(views))}

        images: dict[int, colmap_io.Image] = {}
        for iid, path in enumerate(sorted(images_dir.rglob("*.png")), start=1):
            rel = path.relative_to(images_dir).as_posix()
            view_name = rel.split("/")[0]
            stem = Path(rel).stem.removesuffix(f"_{view_name}")
            view, pose = views[view_name], survey.pano_poses[stem]
            R_tls_from_cam = pose.R @ view.R_scanner_from_cam
            centre_sfm = T_SFM_FROM_TLS.apply(pose.t.reshape(1, 3))[0]
            R_sfm_from_cam = T_SFM_FROM_TLS.R @ R_tls_from_cam
            R_cw = R_sfm_from_cam.T
            images[iid] = colmap_io.Image(
                iid,
                rotmat_to_quat(R_cw),
                -R_cw @ centre_sfm,
                camera_id[view_name],
                rel,
                np.zeros((0, 2)),
                np.zeros(0, dtype=np.int64),
            )

        wall = survey.points_tls
        if len(wall) > max_points:
            wall = wall[np.sort(rng.choice(len(wall), max_points, replace=False))]
        pts_sfm = T_SFM_FROM_TLS.apply(wall)
        colour = wall_colour(wall)
        points = {
            j + 1: colmap_io.Point3D(j + 1, pts_sfm[j], colour[j]) for j in range(len(pts_sfm))
        }
        colmap_io.write_model(colmap_io.ColmapModel(cameras, images, points), work_dir / "sparse/0")
        return SfMRun(sparse_root=work_dir / "sparse", commands=[], log=None)

    return _run


class TunnelDepthRenderer(DepthRenderer):
    """Depth by z-buffering the reference cloud, presenting as the renderer this build ships.

    Promotion requires a manifest naming a registered renderer, which is what stops the
    ``renderer=`` seam from minting evidence on its own; a gate that needs the stages *after*
    the render therefore stands in for it explicitly. What comes out is the true wall, so the
    surface, the sections and the volume downstream are computed on something shaped like a
    tunnel instead of on a sphere — and none of it is a measurement of one.
    """

    name = GsplatDepthRenderer.name
    backend = "gsplat"
    pinned_version = GsplatDepthRenderer.pinned_version

    def __init__(self, points_local: np.ndarray, max_depth_m: float = 30.0) -> None:
        self.points = np.asarray(points_local, dtype=np.float64)
        self.max_depth_m = max_depth_m
        self.checkpoint: Path | None = None

    def version(self) -> str:
        return GsplatDepthRenderer.pinned_version

    def require_available(self) -> None:
        return None

    def settings(self) -> dict:
        return {"render_mode": "ED", "stand_in": True}

    def render(self, checkpoint, cameras, images, expected_step=None):
        self.checkpoint = checkpoint
        self.expected_step = expected_step
        for im in images.values():
            cam = cameras[im.camera_id]
            uv, depth = colmap_io.project(cam.K(), im.world_from_cam.inverse(), self.points)
            vis = colmap_io.visible_mask(uv, depth, cam.width, cam.height) & (
                depth < self.max_depth_m
            )
            out = np.full((cam.height, cam.width), np.nan, np.float32)
            if vis.any():
                cols = uv[vis, 0].astype(int).clip(0, cam.width - 1)
                rows = uv[vis, 1].astype(int).clip(0, cam.height - 1)
                d = depth[vis]
                order = np.argsort(-d)  # far first, so the nearest hit is written last
                out[rows[order], cols[order]] = d[order].astype(np.float32)
            yield im, out
