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

Two upstream options are refused outright rather than supported partially:

* ``normalize_world_space: true``: the contract is BACKEND_INTERNAL == LOCAL_METRIC, and the
  Phase 4 audit (docs/PHASE4_CONTRACT.md §8) found that upstream never persists the transform it
  applies, writes checkpoints and PLYs in the normalised frame with no inverse, and that the
  re-implementation below omits upstream's conditional flip. The units-bearing parts of
  training that normalisation would have conditioned are handled in metres instead (AD-6).
* upstream ``depth_loss``: its depth targets are the COLMAP tracks of the very ``points3D`` that
  ``init_type=sfm`` initialises from (``colmap.py:411-420``). Turned on, it would supervise depth
  with the initialisation. MineGS depth supervision is a separate, verified artifact, consumed
  by the MineGS trainer adapter (``minegs.train.trainers.advanced_gs``). Requesting the
  ``depth_loss`` capability means that, and upstream's flag is never emitted.

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
    "normalize_world_space=true is refused: gsplat v1.5.3 keeps the transform it applies only in "
    "memory (parser.transform), writes checkpoints and PLYs in the normalised frame with no "
    "inverse, and adds a conditional 180-degree flip the MineGS re-implementation does not "
    "reproduce, so the outputs could not be returned to LOCAL_METRIC or rendered from the "
    "dataset poses (docs/PHASE4_CONTRACT.md §8: none of the seven conditions is met). Set "
    "backend_args.normalize_world_space: false; metric-unit training is the supported path."
)
DEPTH_LOSS_REFUSAL = (
    "upstream depth_loss is refused for every profile: gsplat v1.5.3 derives its depth targets "
    "from the COLMAP tracks of the same points3D that init_type=sfm initialises from, so the "
    "depth evidence would be the initialisation itself (docs/PHASE4_CONTRACT.md C0 Q6). Depth "
    "supervision is requests.depth_loss: true with a verified DepthSupervisionRecord "
    "(--depth-supervision), consumed by the MineGS trainer adapter."
)
DEPTH_SUPERVISION_NOTE = (
    "depth_loss is delivered as MineGS depth supervision: a DepthSupervisionRecord built by "
    "`minegs dataset depth-supervision`, verified against the dataset, and passed to the run "
    "with --depth-supervision."
)
NORMAL_LOSS_NOTE = (
    "gsplat v1.5.3 simple_trainer has no normal loss; normal consistency exists only in "
    "simple_trainer_2dgs.py, and 2DGS is outside Phase 4 (docs/PHASE4_CONTRACT.md AD-10)."
)
INIT_RANDOM_REFUSAL = (
    "init_type=random draws a cube around the world origin (simple_trainer.py:238). In "
    "LOCAL_METRIC the origin is wherever the dataset put it, so the cube need not contain the "
    "drift; only initialisation from points (init_type: sfm) is accepted."
)
#: Upstream values the depth renderer assumes the model was trained under. A profile may not
#: change them: the run would train, and its depth would then be refused or, worse, rendered
#: under the wrong projection (docs/PHASE4_CONTRACT.md AD-8, Q9).
RENDERER_ASSUMED = {
    "camera_model": "pinhole",
    "near_plane": 0.01,
    "far_plane": 1e10,
    "with_ut": False,
    "with_eval3d": False,
    "pose_noise": 0.0,
    "patch_size": None,
}
#: The mcmc preset's two units-bearing values, as v1.5.3 sets them (simple_trainer.py:1213-1221,
#: gsplat/strategy/mcmc.py:50). The adapter rescales them for metres (AD-6).
UPSTREAM_MCMC_NOISE_LR = 5e5
UPSTREAM_MCMC_SCALE_REG = 0.01
UPSTREAM_TEST_EVERY = 8
UPSTREAM_SH_DEGREE = 3
ADAPTER_MODULE = "minegs.train.trainers.advanced_gs"
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


def _python_tag(loader, suffix, node):
    """A ``!!python/...`` node read as data: mapping -> dict (tag kept), sequence -> list."""
    import yaml

    if isinstance(node, yaml.MappingNode):
        out = loader.construct_mapping(node, deep=True)
        out["__tag__"] = suffix
        return out
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


def read_trainer_config(path: Path) -> dict:
    """The trainer's ``cfg.yml`` as data, nested strategy included, without executing a tag.

    Upstream writes it with ``yaml.dump(vars(cfg))`` (simple_trainer.py:552-554), whose default
    Dumper tags the strategy ``!!python/object:gsplat.strategy...`` and tuples
    ``!!python/tuple``. ``safe_load`` refuses those, and ``unsafe_load`` would import gsplat to
    rebuild them. This loader turns every python tag into plain data and keeps the tag name.
    """
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)
    try:
        data = yaml.load(Path(path).read_text(), Loader=_Loader)
    except (OSError, yaml.YAMLError) as e:
        raise ContractError(f"{path}: the trainer's config cannot be read ({e})") from e
    if not isinstance(data, dict):
        raise ContractError(f"{path}: the trainer's config is not a mapping")
    return data


def strategy_name(cfg: dict) -> str | None:
    """``DefaultStrategy`` / ``MCMCStrategy`` from the strategy tag, or None."""
    st = cfg.get("strategy")
    if isinstance(st, dict) and isinstance(st.get("__tag__"), str):
        return st["__tag__"].rsplit(".", 1)[-1]
    return None


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
    capability_notes = {
        "depth_loss": DEPTH_SUPERVISION_NOTE,
        "normal_loss": NORMAL_LOSS_NOTE,
        "resume": RESUME_REFUSAL,
    }

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
            # MineGS depth supervision, through the trainer adapter (Phase 4 AD-5). Upstream's
            # own depth_loss flag stays refused (DEPTH_LOSS_REFUSAL).
            depth_loss=True,
            normal_loss=False,  # NORMAL_LOSS_NOTE
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
        depth_supervision_dir: Path | None = None,
        depth_supervision_sha256: str | None = None,
    ) -> TrainCommand:
        """``dataset_dir`` must be the *staged* (writable) dataset; see module docstring.

        ``depth_supervision_dir`` is where the trainer will find the staged, verified depth
        artifact (a container path on the docker route), and ``depth_supervision_sha256`` the
        directory hash the run recorded for it. Both are required exactly when the profile
        requests ``depth_loss``.
        """
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
        if bool(args.pop("depth_loss", False)):
            raise ContractError(DEPTH_LOSS_REFUSAL)
        if str(args.get("init_type", "sfm")) != "sfm":
            raise ContractError(INIT_RANDOM_REFUSAL)
        for key, want in RENDERER_ASSUMED.items():
            if key in args and args[key] != want:
                raise ContractError(
                    f"backend_args {key}={args[key]!r}: the depth renderer reproduces a model "
                    f"trained with {key}={want!r} only, so this run's depth could not be "
                    "rendered as trained (docs/PHASE4_CONTRACT.md AD-8)"
                )
        depth = bool(enabled.get("depth_loss"))
        if depth and (depth_supervision_dir is None or not depth_supervision_sha256):
            raise ContractError(
                "the profile requests depth supervision (requests.depth_loss: true) but no "
                "verified DepthSupervisionRecord was given (--depth-supervision). Fake COLMAP "
                "tracks are not a substitute: upstream's own depth flag is refused."
            )
        if not depth and depth_supervision_dir is not None:
            raise ContractError(
                "a depth supervision artifact was given to a profile that does not request "
                "depth_loss; it would be ignored, and a run that silently ignores its "
                "supervision is recorded as something it is not"
            )
        if not depth and "depth_lambda" in args:
            raise ContractError("backend_args depth_lambda without requests.depth_loss: true")
        if depth and enabled.get("pose_refinement"):
            raise ContractError(
                "depth supervision with pose refinement: the targets are computed from the "
                "dataset poses and the render from refined ones, so they drift apart"
            )
        if enabled.get("bilateral_grid") is not True and bool(args.get("use_fused_bilagrid")):
            raise ContractError(
                "use_fused_bilagrid turns the bilateral grid on (simple_trainer.py:1228-1230); "
                "request bilateral_grid explicitly instead"
            )
        compensate = strategy == "mcmc"
        if compensate:
            for key in ("strategy.noise_lr", "scale_reg"):
                if key in args:
                    raise ContractError(
                        f"backend_args {key}: under mcmc MineGS sets this from the metric scale "
                        "of the cameras (docs/PHASE4_CONTRACT.md AD-6), so a fixed value would "
                        "be in the wrong units"
                    )
        test_every = args.get("test_every", UPSTREAM_TEST_EVERY)
        if not (
            isinstance(test_every, int) and not isinstance(test_every, bool) and test_every >= 1
        ):
            raise ContractError(f"test_every must be a positive integer, got {test_every!r}")
        script = trainer or self._trainer
        if script is None:
            script = locate_trainer(require=check_trainer) or Path(TRAINER_IMAGE_PATH)
        upstream = [
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
        ):  # depth_loss is MineGS supervision, delivered by the adapter, never upstream's flag
            if enabled.get(cap):
                upstream.append(flag)
        if enabled.get("absgrad"):
            if strategy != "default":
                raise ContractError("absgrad is a DefaultStrategy option; not available with mcmc")
            upstream.append("--strategy.absgrad")
        for k, v in args.items():
            if isinstance(v, bool):
                upstream.append(f"--{k}" if v else f"--no-{k}")
            elif isinstance(v, (list, tuple)):
                upstream += [f"--{k}", *[str(x) for x in v]]
            else:
                upstream += [f"--{k}", str(v)]

        if depth or compensate:
            from minegs.train.trainers.advanced_gs import ADAPTER_EVIDENCE, adapter_sha256

            argv = [
                "python",
                "-m",
                ADAPTER_MODULE,
                "--trainer",
                str(script),
                "--evidence",
                str(Path(out_dir) / ADAPTER_EVIDENCE),
            ]
            if depth:
                argv += [
                    "--depth-supervision",
                    str(depth_supervision_dir),
                    "--depth-supervision-sha256",
                    str(depth_supervision_sha256),
                ]
            if compensate:
                argv.append("--mcmc-metric-compensation")
            argv += ["--", *upstream]
            trainer_info = {
                "entrypoint": "minegs_adapter",
                "adapter_module": ADAPTER_MODULE,
                "adapter_sha256": adapter_sha256(),
                "upstream_trainer": str(script),
            }
        else:
            argv = ["python", str(script), *upstream]
            trainer_info = {"entrypoint": "upstream", "upstream_trainer": str(script)}
        _assert_no_refused_flags(argv)

        steps_scaler = args.get("steps_scaler", 1.0)
        expected: dict[str, object] = {
            "normalize_world_space": False,
            "depth_loss": False,
            "data_factor": int(profile.data_factor),
            "max_steps": int(int(profile.max_steps) * float(steps_scaler)),
            "app_opt": bool(enabled.get("appearance_embedding")),
            "use_bilateral_grid": bool(enabled.get("bilateral_grid")),
            "antialiased": bool(enabled.get("antialiasing")),
            "pose_opt": bool(enabled.get("pose_refinement")),
            "init_type": "sfm",
            "sh_degree": args.get("sh_degree", UPSTREAM_SH_DEGREE),
            "test_every": test_every,
            "strategy": "MCMCStrategy" if strategy == "mcmc" else "DefaultStrategy",
            **RENDERER_ASSUMED,
        }
        if strategy == "default":
            expected["strategy.absgrad"] = bool(enabled.get("absgrad"))
        if depth:
            expected["depth_lambda"] = float(args.get("depth_lambda", 0.01))
        # identity by construction: normalisation is refused above (BACKEND_INTERNAL == LOCAL_METRIC)
        return TrainCommand(
            argv=argv,
            env={"MINEGS_BACKEND": self.name},
            T_local_from_internal=Sim3.identity(),
            trainer=trainer_info,
            expected_config=expected,
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
        # The trainer's own account of itself travels with the run, so the render gate and a
        # later reader do not depend on backend_out surviving (Phase 4 C0 §2.3-12).
        from minegs.train.trainers.advanced_gs import ADAPTER_EVIDENCE

        for name in ("cfg.yml", ADAPTER_EVIDENCE):
            if (out_dir / name).is_file():
                (run_dir / "trainer").mkdir(exist_ok=True)
                shutil.copy2(out_dir / name, run_dir / "trainer" / name)
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
                ev.stats_gaussian_count = ev.gaussian_count

        ev.renders = sorted(p for p in out_dir.glob("renders/*") if p.is_file())
        cfg = out_dir / "cfg.yml"
        ev.trainer_config = _trainer_config(cfg)
        if cfg.is_file():
            from minegs.core.provenance import sha256_file

            ev.trainer_config_sha256 = sha256_file(cfg)
            try:
                ev.trainer_config_full = read_trainer_config(cfg)
            except ContractError as e:
                ev.notes.append(str(e))
        from minegs.train.trainers.advanced_gs import ADAPTER_EVIDENCE

        adapter = out_dir / ADAPTER_EVIDENCE
        if adapter.is_file():
            try:
                ev.adapter_evidence = json.loads(adapter.read_text())
            except (OSError, ValueError) as e:
                ev.notes.append(f"{ADAPTER_EVIDENCE} could not be read ({e})")
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
    It is **not** what the v1.5.3 Parser does: this composes ``T2 @ T1`` like
    ``normalize.py::normalize``, while ``Parser.__init__`` adds a conditional 180-degree flip
    ``T3`` (``colmap.py:229-244``). An equivalence test must take its reference from the Parser
    (docs/PHASE4_CONTRACT.md §2.3-7). ``similarity_from_cameras`` itself matches upstream's
    function, and the MCMC metric compensation check relies on that part only.
    """
    model = colmap_io.read_model(sparse_dir)
    c2w = np.stack([im.world_from_cam.matrix() for im in model.images.values()])
    T1 = similarity_from_cameras(c2w)
    pts = model.points_xyz()
    if len(pts) < 3:
        return T1
    T2 = align_principle_axes(T1.apply(pts))
    return Sim3.from_se3(T2) @ T1
