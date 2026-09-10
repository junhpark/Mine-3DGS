"""gsplat adapter (§8.1) — v0.1's only backend.

Executable contract (verified against gsplat v1.5.3 upstream):

* The trainer is ``examples/simple_trainer.py`` in the gsplat *repository*, not a module of
  the PyPI wheel. ``docker/Dockerfile.gpu`` clones the tag matching the wheel to
  ``/opt/gsplat`` and exports ``MINEGS_GSPLAT_TRAINER``; ``build_command`` refuses to build a
  command when that script cannot be located (env var, or the image default path).
* Sub-commands are ``default`` / ``mcmc`` (the densification strategy). ``absgrad`` is a
  field of ``DefaultStrategy`` (``--strategy.absgrad``), not a top-level flag.
* The COLMAP parser *writes* ``images_<factor>_png`` next to ``images/`` when a downscaled
  folder is missing, so the trainer must be pointed at a writable **staged** copy of the
  dataset (``minegs.train.staging``), never at the read-only dataset mount.

Frame duties: we pass ``--no-normalize_world_space`` so BACKEND_INTERNAL == LOCAL_METRIC.
If a profile insists on ``normalize_world_space: true`` we recompute gsplat's normalisation
(``similarity_from_cameras`` + ``align_principle_axes``, re-implemented below) and invert it
on export. Either way ``run.json`` records ``T_local_from_internal``.
"""

from __future__ import annotations

import os
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
TRAINER_ENV = "MINEGS_GSPLAT_TRAINER"
TRAINER_IMAGE_PATH = "/opt/gsplat/examples/simple_trainer.py"  # set in docker/Dockerfile.gpu
STRATEGIES = ("default", "mcmc")


def locate_trainer(require: bool = True) -> Path | None:
    """Path of gsplat's ``examples/simple_trainer.py`` (env override, else image default)."""
    cand = Path(os.environ.get(TRAINER_ENV, TRAINER_IMAGE_PATH))
    if cand.exists():
        return cand
    if require:
        raise ContractError(
            f"gsplat trainer not found at {cand} (the PyPI wheel does not ship it). Run inside "
            f"docker/Dockerfile.gpu, or clone gsplat v{PINNED_GSPLAT} and set {TRAINER_ENV}="
            "<repo>/examples/simple_trainer.py"
        )
    return None


class GsplatBackend(TrainBackend):
    name = "gsplat"

    def __init__(self, trainer: Path | None = None) -> None:
        self._trainer = trainer

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
        self,
        dataset_dir: Path,
        out_dir: Path,
        profile: Profile,
        resume: bool = False,
        trainer: Path | None = None,
        check_trainer: bool = True,
    ) -> TrainCommand:
        """``dataset_dir`` must be the *staged* (writable) dataset; see module docstring."""
        enabled = self.resolve_requests(profile)
        args = dict(profile.backend_args)
        strategy = str(args.pop("strategy", "default"))
        if strategy not in STRATEGIES:
            raise ContractError(f"gsplat strategy must be one of {STRATEGIES}, got {strategy!r}")
        normalize = bool(args.pop("normalize_world_space", False))
        script = trainer or self._trainer
        if script is None:
            script = locate_trainer(require=check_trainer) or Path(TRAINER_IMAGE_PATH)
        argv = [
            "python",
            str(script),
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
            "--disable_viewer",
        ]
        # profile.max_images is applied by minegs.train.staging (image subset written into the
        # staged dataset), never by the trainer.
        for cap, flag in (
            ("appearance_embedding", "--app_opt"),
            ("bilateral_grid", "--use_bilateral_grid"),
            ("depth_loss", "--depth_loss"),
            ("antialiasing", "--antialiased"),
            ("pose_refinement", "--pose_opt"),
        ):
            if enabled.get(cap):
                argv.append(flag)
        if enabled.get("absgrad"):
            if strategy != "default":
                raise ContractError("absgrad is a DefaultStrategy option; not available with mcmc")
            argv.append("--strategy.absgrad")
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
