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

Phase 0A/0D refuses two requests outright rather than supporting them partially:

* ``normalize_world_space: true`` — the baseline contract is BACKEND_INTERNAL == LOCAL_METRIC.
  Our re-implementation of gsplat's normalisation below is *unvalidated* against upstream
  (per-dataset orientation handling is not proven equivalent), and in the docker path the
  command is built on the host from a container-side ``data_dir``, so the transform would be
  computed from a path that does not exist there. Enabling it half-way would silently corrupt
  every metric claim, so it raises ``ContractError``. The helpers stay for the future
  equivalence test; nothing in the command path calls them.
* ``depth_loss: true`` — upstream depth supervision reads COLMAP image→point observation
  tracks, and TLS-initialised staging replaces ``points3D`` with ``init_points.ply`` and
  clears those tracks (``minegs.train.staging``). Depth supervision is redesigned in Phase 4.

A third refusal is forced by the pinned trainer itself: **v1.5.3 cannot resume training.**
``Config.ckpt`` is documented upstream as *"Path to the .pt files. If provide, it will skip
training and run evaluation only."*, and ``main()`` branches on it — ``if cfg.ckpt is not
None:`` runs ``eval``/``render_traj`` and returns, ``else:`` runs ``train()``. ``train()``
sets ``init_step = 0`` unconditionally and never loads a checkpoint, and the saved ``.pt``
holds only ``{"step", "splats"}`` (plus pose/appearance modules) — no optimizer moments and
no densification-strategy state. So there is no combination of upstream flags that continues
a run: passing ``--ckpt`` alongside training arguments produces an *evaluation* pass on the
parent's weights, which is neither the requested experiment nor a visible failure. This
adapter therefore declares ``resume=False``, and ``--resume-from`` is refused before a command
is built. Resuming would need a trainer and checkpoint that restore the whole training state
(Phase 0D.3, docs/ROADMAP.md); Phase 0D.2 runs its baseline uninterrupted instead.

``T_local_from_internal`` is therefore identity on every command this adapter builds, and
``run.json`` records it explicitly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Sim3
from minegs.core.pointcloud import read_ply, write_ply
from minegs.ingest.common import colmap_io
from minegs.train.backends.base import (
    BackendCapabilities,
    TrainBackend,
    TrainCommand,
    TrainEvidence,
)
from minegs.train.profiles import Profile

PINNED_GSPLAT = "1.5.3"
TRAINER_ENV = "MINEGS_GSPLAT_TRAINER"
TRAINER_IMAGE_PATH = "/opt/gsplat/examples/simple_trainer.py"  # set in docker/Dockerfile.gpu
STRATEGIES = ("default", "mcmc")

NORMALIZE_REFUSAL = (
    "normalize_world_space=true is not enabled in the Phase 0A/0D baseline. Mine-3DGS requires "
    "BACKEND_INTERNAL == LOCAL_METRIC until upstream transform equivalence is validated "
    "(docs/ROADMAP.md). Set backend_args.normalize_world_space: false."
)
DEPTH_LOSS_REFUSAL = (
    "depth_loss requested, but TLS initialization staging removes the COLMAP observation "
    "tracks required by upstream gsplat depth supervision (minegs.train.staging writes "
    "init_points.ply as points3D and clears image point3D_ids). This capability is deferred "
    "to Phase 4 (docs/ROADMAP.md); set requests.depth_loss: false to run this profile."
)
RESUME_REFUSAL = (
    f"gsplat v{PINNED_GSPLAT}'s examples/simple_trainer.py cannot continue training from a "
    "checkpoint: --ckpt is documented as 'If provide, it will skip training and run evaluation "
    "only', main() runs eval instead of train() whenever it is set, train() starts at "
    "init_step = 0 unconditionally, and the saved .pt carries only step and splats (no "
    "optimizer or densification-strategy state). Passing --ckpt to a training run would "
    "silently produce an evaluation pass on the parent's weights rather than a continuation, "
    "so this adapter refuses resume instead of appearing to support it. Resuming would need a "
    "trainer and checkpoint that restore the whole training state — optimizer, schedulers, "
    "strategy state, step, RNG (Phase 0D.3, docs/ROADMAP.md)."
)


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


# Options whose *enabled* form must never appear in an assembled command, whatever route
# (capability request, raw backend_args, future edit) tried to put it there. ``ckpt`` is here
# because a backend_args entry would otherwise be forwarded verbatim by the passthrough loop
# and turn the run into an evaluation pass (see RESUME_REFUSAL) with no error anywhere.
REFUSED_FLAGS = {
    "depth_loss": DEPTH_LOSS_REFUSAL,
    "normalize_world_space": NORMALIZE_REFUSAL,
    "ckpt": RESUME_REFUSAL,
}


#: ``ckpt_6999_rank0.pt`` -> 6999, ``point_cloud_6999.ply`` -> 6999, ``train_step6999_rank0``
#: -> 6999. The first run of digits after the last underscore-delimited word that starts one.
_STEP_RE = re.compile(r"(?:step|ckpt|point_cloud)[_]?(\d+)")


def _step_of(name: str) -> int | None:
    m = _STEP_RE.search(name)
    return int(m.group(1)) if m else None


def _as_float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


#: ``key: value`` at the top level of the trainer's own ``cfg.yml``. Read with a regex rather
#: than a YAML parser on purpose: upstream writes it with ``yaml.dump(vars(cfg))`` (v1.5.3
#: simple_trainer.py:552-554), whose default Dumper tags ``strategy`` as
#: ``!!python/object:gsplat.strategy...``. ``safe_load`` refuses that, and ``unsafe_load`` would
#: import gsplat to reconstruct it — executing trainer output to read a flag is not a trade this
#: check needs to make. Nested fields are indented, so this sees only the top level.
_CFG_LINE = re.compile(r"(?m)^([A-Za-z_]\w*):[ \t]+(.+?)[ \t]*$")


def _trainer_config(path: Path) -> dict[str, str]:
    """What the trainer recorded about its own configuration, as plain strings."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    return {m.group(1): m.group(2) for m in _CFG_LINE.finditer(text)}


def canonical_option(raw: object) -> str:
    """One spelling per option, for a ``backend_args`` key and an assembled token alike.

    A refusal that matches one spelling of a flag is not a refusal. tyro (gsplat's CLI parser)
    accepts ``--depth-loss`` and ``--depth_loss`` alike, so hyphens fold to underscores; and a
    key written with its dashes already attached — ``backend_args: {"--ckpt": ...}`` — used to be
    emitted as ``--__ckpt`` and sail past a guard keyed on ``ckpt``, so leading dashes come off
    too. Surrounding whitespace is stripped for the same reason: ``{"ckpt ": ...}`` rendered as
    ``--ckpt <path>`` in the printed command while matching nothing. Leading underscores go the
    same way, so the guard still recognises a token like ``--__ckpt`` however it was produced; no
    gsplat option name begins with one.
    """
    return str(raw).strip(" \t\r\n-_").replace("-", "_")


def _assert_no_refused_flags(argv: list[str]) -> None:
    """Last line of defence: scan the assembled argv, not just the inputs that built it."""
    for token in argv:
        if not token.startswith("-"):
            continue
        name = canonical_option(token.split("=", 1)[0])
        if name.startswith("no_"):  # --no-<opt> disables it; that is the safe direction
            continue
        # Case-folded because REFUSED_FLAGS is lower-case and no gsplat option is not: matching
        # only the exact case would let ``--CKPT`` through a guard whose whole job is to be the
        # spelling-independent one.
        if name.casefold() in REFUSED_FLAGS:
            raise ContractError(REFUSED_FLAGS[name.casefold()])


class GsplatBackend(TrainBackend):
    name = "gsplat"
    capability_notes = {"depth_loss": DEPTH_LOSS_REFUSAL, "resume": RESUME_REFUSAL}

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
            # upstream gsplat supports depth supervision, but this adapter cannot deliver it
            # while staging replaces points3D with TLS points (see DEPTH_LOSS_REFUSAL).
            depth_loss=False,
            normal_loss=False,
            antialiasing=True,
            absgrad=True,
            mcmc_strategy=True,
            pose_refinement=True,
            depth_render=True,
            # v1.5.3 has no resume path at all; see RESUME_REFUSAL and the module docstring.
            resume=False,
        )

    def build_command(
        self,
        dataset_dir: Path,
        out_dir: Path,
        profile: Profile,
        trainer: Path | None = None,
        check_trainer: bool = True,
    ) -> TrainCommand:
        """``dataset_dir`` must be the *staged* (writable) dataset; see module docstring."""
        enabled = self.resolve_requests(profile)
        # Canonicalise every backend_args key first, so the refusals below and the argv scan at
        # the end are comparing the same thing the user wrote (see ``canonical_option``): one
        # key, one guard. Without this a refused option smuggled in under a second spelling is
        # forwarded verbatim by the passthrough loop.
        args: dict[str, object] = {}
        for raw_key, value in profile.backend_args.items():
            key = canonical_option(raw_key)
            if not key or any(c.isspace() for c in key):
                raise ContractError(
                    f"backend_args key {raw_key!r} is not an option name. A key with inner "
                    "whitespace cannot be passed as a flag, and a command printed with one reads "
                    "as two arguments."
                )
            if key in args:
                raise ContractError(
                    f"backend_args has two spellings of the same option ({raw_key!r} collides with "
                    f"{key!r}); keep one"
                )
            args[key] = value
        strategy = str(args.pop("strategy", "default"))
        if strategy not in STRATEGIES:
            raise ContractError(f"gsplat strategy must be one of {STRATEGIES}, got {strategy!r}")
        # Refusals cover both routes into a flag: a profile capability request, and a raw
        # backend_args override that would otherwise be forwarded verbatim below.
        if bool(args.pop("normalize_world_space", False)):
            raise ContractError(NORMALIZE_REFUSAL)
        if enabled.get("depth_loss") or bool(args.pop("depth_loss", False)):
            raise ContractError(DEPTH_LOSS_REFUSAL)
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
            "--no-normalize_world_space",  # BACKEND_INTERNAL == LOCAL_METRIC (§3), enforced above
            "--disable_viewer",
        ]
        # profile.max_images is applied by minegs.train.staging (image subset written into the
        # staged dataset), never by the trainer.
        for cap, flag in (
            ("appearance_embedding", "--app_opt"),
            ("bilateral_grid", "--use_bilateral_grid"),
            ("antialiasing", "--antialiased"),
            ("pose_refinement", "--pose_opt"),
        ):  # depth_loss is refused above, never emitted
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
        _assert_no_refused_flags(argv)
        # identity by construction: normalisation is refused above (BACKEND_INTERNAL == LOCAL_METRIC)
        return TrainCommand(
            argv=argv,
            env={"MINEGS_BACKEND": self.name},
            T_local_from_internal=Sim3.identity(),
        )

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

    def collect_evidence(self, out_dir: Path, profile: Profile) -> TrainEvidence:
        """Read back what ``simple_trainer.py`` wrote (§0D.2 D2-4, D2-5, D2-6, D2-9).

        Layout and filenames are taken from upstream v1.5.3 itself (sha256
        ``79319e1c…62c05``), not guessed: ``result_dir`` gains ``ckpts/``, ``stats/``,
        ``renders/`` and ``ply/`` (lines 321-328), the checkpoint is
        ``ckpts/ckpt_{step}_rank{n}.pt`` and the PLY ``ply/point_cloud_{step}.ply``.

        The step numbers are the trap. Both are written at ``step == max_steps - 1``, so a
        7000-step run ends at step 6999 and a check expecting 7000 would fail every real run.
        ``stats/train_step{step:04d}_rank{n}.json`` carries ``{"mem", "ellipse_time",
        "num_GS"}`` — ``mem`` being ``torch.cuda.max_memory_allocated()`` in GiB, measured
        inside the trainer, which is the only honest source for this run's peak GPU memory.
        """
        ev = TrainEvidence(configured_max_steps=profile.max_steps)

        ev.checkpoints = sorted(out_dir.glob("ckpts/ckpt_*.pt"))
        if ev.checkpoints:
            by_step = sorted(ev.checkpoints, key=lambda p: (_step_of(p.name) or -1, p.name))
            ev.final_checkpoint = by_step[-1]
            ev.checkpoint_step = _step_of(ev.final_checkpoint.name)

        plys = sorted(out_dir.glob("ply/*.ply")) + sorted(out_dir.glob("point_cloud/*.ply"))
        if plys:
            ev.final_model = max(plys, key=lambda p: (_step_of(p.name) or -1, p.name))

        stats = sorted(out_dir.glob("stats/train_step*.json"))
        steps = [s for s in (_step_of(p.name) for p in stats) if s is not None]
        if steps:
            ev.observed_final_step = max(steps)
        # The checkpoint's own step is the stronger witness: it is written by the same branch
        # that saves the weights, so it cannot outrun what was actually trained.
        if ev.checkpoint_step is not None:
            ev.observed_final_step = max(ev.observed_final_step or 0, ev.checkpoint_step)

        if stats:
            last = max(stats, key=lambda p: (_step_of(p.name) or -1, p.name))
            try:
                blob = json.loads(last.read_text())
            except (OSError, ValueError) as e:
                ev.notes.append(f"{last.name} could not be read ({e})")
            else:
                ev.peak_gpu_memory_gb = _as_float(blob.get("mem"))
                ev.train_seconds = _as_float(blob.get("ellipse_time"))
                num_gs = blob.get("num_GS")
                ev.gaussian_count = int(num_gs) if isinstance(num_gs, (int, float)) else None

        ev.renders = sorted(p for p in out_dir.glob("renders/*") if p.is_file())
        ev.trainer_config = _trainer_config(out_dir / "cfg.yml")
        return ev


# ------------------------------------------------- gsplat normalisation (FUTURE WORK, unused)
#
# Kept for the upstream-equivalence test that must pass before normalize_world_space=true can
# be enabled (see NORMALIZE_REFUSAL). No code path in this module calls gsplat_normalization;
# it is exercised only by its own unit test.


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
    """``T_internal_from_local`` that gsplat's COLMAP parser applies with normalize=True.

    FUTURE WORK — not equivalence-tested against upstream and not reachable from
    ``build_command``; ``normalize_world_space=true`` is refused (see ``NORMALIZE_REFUSAL``).
    """
    model = colmap_io.read_model(sparse_dir)
    c2w = np.stack([im.world_from_cam.matrix() for im in model.images.values()])
    T1 = similarity_from_cameras(c2w)
    pts = model.points_xyz()
    if len(pts) < 3:
        return T1
    T2 = align_principle_axes(T1.apply(pts))
    return Sim3.from_se3(T2) @ T1
