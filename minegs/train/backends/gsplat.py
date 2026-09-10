"""gsplat adapter (§8.1) — v0.1's only backend.

Phase 0: a pinned wrapper around ``gsplat``'s ``simple_trainer.py``; long-term a thin
``MineGSTrainer`` on the gsplat library. Two frame-related duties:

1. The dataset's ``sparse/0`` is already LOCAL_METRIC. We pass ``--no-normalize_world_space``
   so BACKEND_INTERNAL == LOCAL_METRIC and the exported PLY needs no inverse transform.
2. If a profile insists on ``normalize_world_space: true`` we recompute the exact
   normalisation gsplat applies (``similarity_from_cameras`` + ``align_principle_axes``,
   re-implemented in ``gsplat_normalization``) and invert it on export. Either way
   ``run.json`` records ``T_local_from_internal``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Sim3
from minegs.core.pointcloud import read_ply, write_ply
from minegs.ingest.common import colmap_io
from minegs.train.backends.base import BackendCapabilities, TrainBackend, TrainCommand
from minegs.train.profiles import Profile

PINNED_GSPLAT = "1.5.3"


class GsplatBackend(TrainBackend):
    name = "gsplat"

    def version(self) -> str:
        try:
            import gsplat

            return str(gsplat.__version__)
        except ImportError:
            return f"{PINNED_GSPLAT} (pinned, not installed here)"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            appearance_embedding=True,
            bilateral_grid=True,
            depth_loss=True,
            normal_loss=False,
            antialiasing=True,
            absgrad=True,
            mcmc_strategy=True,
            pose_refinement=True,
            depth_render=True,
            resume=True,
        )

    def build_command(
        self, dataset_dir: Path, out_dir: Path, profile: Profile, resume: bool = False
    ) -> TrainCommand:
        enabled = self.resolve_requests(profile)
        args = dict(profile.backend_args)
        strategy = args.pop("strategy", "default")
        normalize = bool(args.pop("normalize_world_space", False))
        argv = [
            "python",
            "-m",
            "gsplat.examples.simple_trainer",
            strategy,
            "--data_dir",
            str(dataset_dir),
            "--result_dir",
            str(out_dir),
            "--data_factor",
            str(profile.data_factor),
            "--max_steps",
            str(profile.max_steps),
            "--normalize_world_space" if normalize else "--no-normalize_world_space",
        ]
        # profile.max_images is *not* a gsplat flag: the light profile's image subset is
        # produced by writing a reduced sparse/0 (Phase 0D, runner side), never by the trainer.
        for cap, flag in (
            ("appearance_embedding", "--app_opt"),
            ("bilateral_grid", "--use_bilateral_grid"),
            ("depth_loss", "--depth_loss"),
            ("antialiasing", "--antialiased"),
            ("absgrad", "--absgrad"),
            ("pose_refinement", "--pose_opt"),
        ):
            if enabled.get(cap):
                argv.append(flag)
        for k, v in args.items():
            if isinstance(v, bool):
                argv.append(f"--{k}" if v else f"--no-{k}")
            elif isinstance(v, (list, tuple)):
                argv += [f"--{k}", *[str(x) for x in v]]
            else:
                argv += [f"--{k}", str(v)]
        if resume:
            ck = (
                sorted((out_dir / "ckpts").glob("ckpt_*.pt"))
                if (out_dir / "ckpts").exists()
                else []
            )
            if ck:
                argv += ["--ckpt", str(ck[-1])]
        T = Sim3.identity()
        if normalize:
            T = gsplat_normalization(dataset_dir / "sparse" / "0").inverse()
        return TrainCommand(argv=argv, env={"MINEGS_BACKEND": self.name}, T_local_from_internal=T)

    def normalize_outputs(
        self, out_dir: Path, run_dir: Path, T_local_from_internal: Sim3
    ) -> list[Path]:
        pc_dir = run_dir / "point_cloud"
        pc_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        plys = sorted(out_dir.glob("ply/*.ply")) + sorted(out_dir.glob("point_cloud/*.ply"))
        if not plys:
            raise ContractError(f"{out_dir}: backend produced no PLY (enable save_ply)")
        for p in plys:
            pc = read_ply(p)
            pc.frame = "BACKEND_INTERNAL"
            out = pc.transformed(T_local_from_internal, "LOCAL_METRIC")
            produced.append(write_ply(out, pc_dir / p.name))
        for sub in ("ckpts", "stats", "renders"):
            if (out_dir / sub).exists():
                dst = run_dir / ("ckpt" if sub == "ckpts" else sub)
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(out_dir / sub, dst)
        return produced


# ---------------------------------------------------------------- gsplat normalisation


def similarity_from_cameras(
    c2w: np.ndarray, strict_scaling: bool = False, center_method: str = "focus"
) -> Sim3:
    """Re-implementation of gsplat ``datasets/normalize.py::similarity_from_cameras`` (numpy)."""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(center_method)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return Sim3.from_matrix(transform)


def align_principle_axes(point_cloud: np.ndarray) -> SE3:
    """Re-implementation of gsplat ``datasets/normalize.py::align_principle_axes``."""
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    cov = np.cov(translated, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    rotation = eigvecs.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return SE3.from_matrix(transform)


def gsplat_normalization(sparse_dir: Path) -> Sim3:
    """``T_internal_from_local`` that gsplat's COLMAP parser applies with normalize=True."""
    model = colmap_io.read_model(sparse_dir)
    c2w = np.stack([im.world_from_cam.matrix() for im in model.images.values()])
    T1 = similarity_from_cameras(c2w)
    pts = model.points_xyz()
    if len(pts) < 3:
        return T1
    T2 = align_principle_axes(T1.apply(pts))
    return Sim3.from_se3(T2) @ T1
