"""MineGS advanced trainer adapter (Phase 4 AD-5): upstream ``simple_trainer.py``, plus what
upstream cannot do for a metric survey.

Run as::

    python -m minegs.train.trainers.advanced_gs --trainer <simple_trainer.py> --evidence <json>
        [--depth-supervision <dir> --depth-supervision-sha256 <hex>] [--mcmc-metric-compensation]
        -- <upstream sub-command and flags, exactly as GsplatBackend builds them>

What this is not: a copy of the upstream training loop. The pinned trainer runs as the file it
is, through ``runpy``, with its own tyro CLI, presets, ``adjust_steps`` and bilateral-grid
imports. Upstream launches every run through ``gsplat.distributed.cli(main, cfg)``. Before the
trainer starts, that function is wrapped, so the wrapper receives the fully parsed ``cfg`` and
upstream's own ``main``. It then swaps the ``Runner`` name in ``main``'s globals for a subclass
that changes exactly two things:

* **``__init__``**: after upstream has built its parser, train split and splats, it loads the
  depth supervision (if any) onto upstream's own train indices. With MCMC it rescales the two
  MCMC parameters that carry length units (``noise_lr``, ``scale_reg``), using the scale upstream
  normalisation would have used (AD-6).
* **``rasterize_splats``**: on a training call only, it renders ``RGB+ED``, computes the MineGS
  depth term from the ED channel and attaches its gradient to the returned colours
  (``depth_term.add_gradient``).

Upstream ``depth_loss`` stays false. Its depth targets are the init points themselves (C0 Q6),
and that flag is refused for every run. This module writes ``minegs_trainer.json``, and the
runner compares it with what the run was asked to do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_EVIDENCE = "minegs_trainer.json"
EVIDENCE_SCHEMA = "1.0"
MODULE = "minegs.train.trainers.advanced_gs"


class AdapterError(RuntimeError):
    """A refusal inside the trainer process. It exits non-zero, so the run is FAILED."""


def adapter_sha256() -> str:
    """Digest of the adapter source that would run: this file and the depth term beside it."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("advanced_gs.py", "depth_term.py"):
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- arguments


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise AdapterError(
            "the upstream arguments must follow a literal '--', so nothing meant for upstream "
            "is read as an adapter option and nothing meant for the adapter reaches upstream"
        )
    cut = argv.index("--")
    own, upstream = argv[:cut], argv[cut + 1 :]
    p = argparse.ArgumentParser(prog=f"python -m {MODULE}")
    p.add_argument("--trainer", required=True, help="upstream examples/simple_trainer.py")
    p.add_argument("--evidence", required=True, help="where to write minegs_trainer.json")
    p.add_argument("--depth-supervision", default=None)
    p.add_argument("--depth-supervision-sha256", default=None)
    p.add_argument("--mcmc-metric-compensation", action="store_true")
    ns = p.parse_args(own)
    if (ns.depth_supervision is None) != (ns.depth_supervision_sha256 is None):
        raise AdapterError("--depth-supervision and --depth-supervision-sha256 go together")
    if not upstream or upstream[0] not in ("default", "mcmc"):
        raise AdapterError(f"upstream arguments must start with default|mcmc, got {upstream[:1]}")
    return ns, upstream


# ---------------------------------------------------------------- supervision


@dataclass
class SupervisionTable:
    path: Path
    artifact_sha256: str
    supervision_id: str
    source_kind: str
    confidence_semantics: str
    n_samples: int
    by_name: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    @property
    def n_in_loss(self) -> int:
        return int(sum(len(v[3]) for v in self.by_name.values()))


def load_supervision(path: str | Path, expected_sha256: str) -> SupervisionTable:
    """Read the artifact the host verified, and prove it is the same bytes.

    The full re-derivation (leakage, binding) ran on the host before the command was built
    (``verify_depth_supervision``); inside the container the dataset tree is the staged copy and
    the evidence that matters is identity: the directory hash the run recorded.
    """
    from minegs.core.provenance import sha256_tree
    from minegs.train.supervision.depth import (
        CONFIDENCE_SEMANTICS,
        DEPTH_SEMANTICS,
        DEPTH_UNIT,
        FRAME,
        PIXEL_CONVENTION,
        RECORD_FILE,
        SAMPLE_DTYPE,
    )

    art = Path(path)
    got = sha256_tree(art)
    if got != expected_sha256:
        raise AdapterError(
            f"depth supervision at {art} hashes to {got[:12]}, the run recorded "
            f"{expected_sha256[:12]}: these are not the samples that were verified"
        )
    rec = json.loads((art / RECORD_FILE).read_text())
    for k, want in (
        ("frame", FRAME),
        ("depth_unit", DEPTH_UNIT),
        ("depth_semantics", DEPTH_SEMANTICS),
        ("pixel_convention", PIXEL_CONVENTION),
    ):
        if rec.get(k) != want:
            raise AdapterError(f"depth supervision {k}={rec.get(k)!r}, expected {want!r}")
    if rec.get("confidence_semantics") not in CONFIDENCE_SEMANTICS:
        raise AdapterError(f"unknown confidence_semantics {rec.get('confidence_semantics')!r}")
    samples_path = art / rec["samples_file"]
    if _sha256_file(samples_path) != rec["samples_sha256"]:
        raise AdapterError("samples.npy does not match its record")
    s = np.load(samples_path, allow_pickle=False)
    if s.dtype != SAMPLE_DTYPE:
        raise AdapterError(f"samples dtype {s.dtype} is not {SAMPLE_DTYPE}")
    table = SupervisionTable(
        path=art,
        artifact_sha256=got,
        supervision_id=rec["supervision_id"],
        source_kind=rec["source_kind"],
        confidence_semantics=rec["confidence_semantics"],
        n_samples=len(s),
    )
    names = rec["images"]
    keep = s["confidence"] > 0  # confidence 0 is recorded for audit and never trains (AD-4)
    s = s[keep]
    for k in np.unique(s["image"]):
        sel = s[s["image"] == k]
        table.by_name[names[int(k)]] = (
            sel["u"].astype(np.float64),
            sel["v"].astype(np.float64),
            sel["depth_m"].astype(np.float64),
            sel["confidence"].astype(np.float64),
        )
    return table


def full_resolution_K(data_dir: str | Path) -> dict[int, np.ndarray]:
    """The dataset cameras' K at full resolution, from the staged text model (MineGS reader)."""
    from minegs.ingest.common import colmap_io

    model = colmap_io.read_model(Path(data_dir) / "sparse" / "0")
    return {int(c.id): c.K() for c in model.cameras.values()}


def metric_scale_from_cameras(camtoworlds: np.ndarray, similarity_from_cameras: Any) -> float:
    """The scale upstream normalisation would apply to these cameras (``T1``'s ``1/median``).

    ``similarity_from_cameras`` is passed in so the trainer side uses upstream's own function and
    the host side its port; the runner compares the two numbers.
    """
    T = similarity_from_cameras(np.asarray(camtoworlds, dtype=np.float64))
    if hasattr(T, "s"):  # the MineGS port returns a Sim3
        return float(T.s)
    # upstream returns a 4x4 whose top three rows were multiplied by the scale: every row of
    # the rotation block has that norm (also in upstream's det -1 fallback)
    return float(np.linalg.norm(np.asarray(T, dtype=np.float64)[0, :3]))


# ---------------------------------------------------------------- the Runner subclass


@dataclass
class AdapterState:
    """Everything the evidence file reports; filled in by the Runner subclass and the hook."""

    hook_calls: int = 0
    runner_built: bool = False
    train_finished: bool = False
    optimised_images: list[str] = field(default_factory=list)
    images_with_samples: list[str] = field(default_factory=list)
    images_without_samples: list[str] = field(default_factory=list)
    samples_loaded_in_loss: int = 0
    samples_in_domain: int = 0
    samples_out_of_domain: int = 0
    training_calls: int = 0
    steps_with_depth_term: int = 0
    depth_term_sum: float = 0.0
    depth_term_last: float | None = None
    scene_scale: float | None = None
    depth_lambda: float | None = None
    compensation: dict[str, Any] | None = None
    upstream_cfg: dict[str, Any] = field(default_factory=dict)


def make_runner_class(
    base: type, table: SupervisionTable | None, compensate: bool, state: AdapterState
) -> type:
    """Subclass upstream's Runner. Only ``__init__`` and ``rasterize_splats`` are touched."""

    class MineGSRunner(base):  # type: ignore[misc, valid-type]
        def __init__(self, local_rank, world_rank, world_size, cfg):
            if world_size != 1:
                raise AdapterError(
                    f"world_size={world_size}: MineGS trains on exactly one GPU (§0D.2 B2)"
                )
            super().__init__(local_rank, world_rank, world_size, cfg)
            state.runner_built = True
            state.scene_scale = float(self.scene_scale)
            state.optimised_images = [
                self.parser.image_names[int(i)] for i in self.trainset.indices
            ]
            if compensate:
                self._minegs_compensate(cfg)
            self._minegs_samples: dict[int, tuple] = {}
            self._minegs_device_samples: dict[int, tuple] = {}
            if table is not None:
                self._minegs_load(cfg, table)

        def _minegs_compensate(self, cfg) -> None:
            from datasets.normalize import similarity_from_cameras  # upstream's own

            s = metric_scale_from_cameras(self.parser.camtoworlds, similarity_from_cameras)
            base_noise = float(cfg.strategy.noise_lr)
            base_reg = float(cfg.scale_reg)
            cfg.strategy.noise_lr = base_noise * s * s
            cfg.scale_reg = base_reg * s
            state.compensation = {
                "s": s,
                "noise_lr_base": base_noise,
                "noise_lr": float(cfg.strategy.noise_lr),
                "scale_reg_base": base_reg,
                "scale_reg": float(cfg.scale_reg),
                "cameras": len(self.parser.camtoworlds),
                "function": "examples/datasets/normalize.py::similarity_from_cameras",
            }

        def _minegs_load(self, cfg, table: SupervisionTable) -> None:
            from minegs.train.trainers.depth_term import in_domain, to_training_index

            if getattr(cfg, "patch_size", None) is not None:
                raise AdapterError("patch_size crops the image and K; depth samples would drift")
            if getattr(cfg, "pose_opt", False) or getattr(cfg, "pose_noise", 0.0):
                raise AdapterError(
                    "pose refinement/noise renders from poses the depth targets were not "
                    "computed from"
                )
            full_K = full_resolution_K(cfg.data_dir)
            state.depth_lambda = float(cfg.depth_lambda)
            for item, index in enumerate(self.trainset.indices):
                name = self.parser.image_names[int(index)]
                s = table.by_name.get(name)
                if s is None:
                    state.images_without_samples.append(name)
                    continue
                cam = self.parser.camera_ids[int(index)]
                W, H = self.parser.imsize_dict[cam]
                ui, vi = to_training_index(s[0], s[1], full_K[int(cam)], self.parser.Ks_dict[cam])
                keep = in_domain(ui, vi, W, H)
                state.samples_loaded_in_loss += len(keep)
                state.samples_in_domain += int(keep.sum())
                state.samples_out_of_domain += int((~keep).sum())
                if keep.any():
                    self._minegs_samples[item] = (ui[keep], vi[keep], s[2][keep], s[3][keep])
                    state.images_with_samples.append(name)
                else:
                    state.images_without_samples.append(name)

        def _minegs_batch(self, image_ids, device):
            import torch

            out = []
            for item in image_ids.reshape(-1).tolist():
                if item not in self._minegs_samples:
                    out.append(None)
                    continue
                if item not in self._minegs_device_samples:
                    self._minegs_device_samples[item] = tuple(
                        torch.as_tensor(a, dtype=torch.float32, device=device)
                        for a in self._minegs_samples[item]
                    )
                out.append(self._minegs_device_samples[item])
            return out

        def rasterize_splats(
            self,
            camtoworlds,
            Ks,
            width,
            height,
            masks=None,
            rasterize_mode=None,
            camera_model=None,
            **kwargs,
        ):
            import torch

            image_ids = kwargs.get("image_ids")
            training = image_ids is not None and torch.is_grad_enabled()
            if training:
                state.training_calls += 1
            if not (training and table is not None):
                return super().rasterize_splats(
                    camtoworlds,
                    Ks,
                    width,
                    height,
                    masks=masks,
                    rasterize_mode=rasterize_mode,
                    camera_model=camera_model,
                    **kwargs,
                )
            from minegs.train.trainers.depth_term import add_gradient, depth_term_torch

            kwargs["render_mode"] = "RGB+ED"
            renders, alphas, info = super().rasterize_splats(
                camtoworlds,
                Ks,
                width,
                height,
                masks=masks,
                rasterize_mode=rasterize_mode,
                camera_model=camera_model,
                **kwargs,
            )
            colors, ed = renders[..., 0:3], renders[..., 3:4]
            batch = self._minegs_batch(image_ids, ed.device)
            term, n = depth_term_torch(
                ed, batch, float(self.scene_scale), float(self.cfg.depth_lambda)
            )
            if n:
                state.steps_with_depth_term += 1
                value = float(term.detach())
                state.depth_term_sum += value
                state.depth_term_last = value
            return add_gradient(colors, term), alphas, info

    MineGSRunner.__name__ = "MineGSRunner"
    MineGSRunner.__qualname__ = "MineGSRunner"
    return MineGSRunner


# ---------------------------------------------------------------- launch


def _cfg_summary(cfg: Any) -> dict[str, Any]:
    keys = (
        "data_factor",
        "max_steps",
        "normalize_world_space",
        "depth_loss",
        "depth_lambda",
        "app_opt",
        "use_bilateral_grid",
        "antialiased",
        "pose_opt",
        "test_every",
        "sh_degree",
        "init_type",
        "scale_reg",
        "opacity_reg",
        "batch_size",
        "patch_size",
    )
    out = {k: getattr(cfg, k, None) for k in keys}
    out["strategy"] = type(cfg.strategy).__name__
    out["strategy.noise_lr"] = getattr(cfg.strategy, "noise_lr", None)
    return out


def run(argv: list[str]) -> dict[str, Any]:
    ns, upstream = parse_args(argv)
    trainer = Path(ns.trainer).resolve()
    if not trainer.is_file():
        raise AdapterError(f"upstream trainer {trainer} not found")
    examples = trainer.parent
    table = (
        load_supervision(ns.depth_supervision, ns.depth_supervision_sha256)
        if ns.depth_supervision
        else None
    )
    compensate = bool(ns.mcmc_metric_compensation)
    if compensate and upstream[0] != "mcmc":
        raise AdapterError("--mcmc-metric-compensation applies to the mcmc strategy only")
    state = AdapterState()
    started = time.time()

    sys.path.insert(0, str(examples))
    import gsplat.distributed as gd

    upstream_cli = gd.cli

    def hooked_cli(fn, cfg, verbose: bool = False):
        state.hook_calls += 1
        if getattr(cfg, "depth_loss", False):
            raise AdapterError(
                "upstream depth_loss is on. Its targets are the init points3D themselves "
                "(C0 Q6); MineGS depth comes only from the declared artifact"
            )
        if getattr(cfg, "normalize_world_space", True):
            raise AdapterError(
                "normalize_world_space must be false (BACKEND_INTERNAL == LOCAL_METRIC)"
            )
        g = fn.__globals__
        if "Runner" not in g:
            raise AdapterError("the upstream trainer has no Runner to extend; not v1.5.3's layout")
        g["Runner"] = make_runner_class(g["Runner"], table, compensate, state)
        result = upstream_cli(fn, cfg, verbose=verbose)
        state.train_finished = True
        state.upstream_cfg = _cfg_summary(cfg)
        return result

    gd.cli = hooked_cli
    old_argv = sys.argv
    sys.argv = [str(trainer), *upstream]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    finally:
        sys.argv = old_argv
        gd.cli = upstream_cli
    if state.hook_calls != 1 or not state.runner_built or not state.train_finished:
        raise AdapterError(
            f"the upstream trainer did not run through gsplat.distributed.cli once "
            f"(calls={state.hook_calls}, runner_built={state.runner_built}, "
            f"finished={state.train_finished}); the MineGS extensions were never attached"
        )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "adapter": {
            "module": MODULE,
            "sha256": adapter_sha256(),
        },
        "upstream_trainer": {"path": str(trainer), "sha256": _sha256_file(trainer)},
        "upstream_argv": upstream,
        "upstream_cfg": state.upstream_cfg,
        "seconds": time.time() - started,
        "scene_scale": state.scene_scale,
        "optimised_images": state.optimised_images,
        "mcmc_metric_compensation": state.compensation,
        "depth_supervision": None,
    }
    if table is not None:
        from minegs.train.trainers.depth_term import FORMULA

        evidence["depth_supervision"] = {
            "supervision_id": table.supervision_id,
            "artifact_sha256": table.artifact_sha256,
            "source_kind": table.source_kind,
            "confidence_semantics": table.confidence_semantics,
            "n_samples": table.n_samples,
            "n_samples_in_loss_loaded": state.samples_loaded_in_loss,
            "n_samples_in_domain": state.samples_in_domain,
            "n_samples_out_of_domain": state.samples_out_of_domain,
            "images_with_samples": state.images_with_samples,
            "images_without_samples": state.images_without_samples,
            "training_calls": state.training_calls,
            "steps_with_depth_term": state.steps_with_depth_term,
            "depth_term_mean": (state.depth_term_sum / state.steps_with_depth_term)
            if state.steps_with_depth_term
            else None,
            "depth_term_last": state.depth_term_last,
            "depth_lambda": state.depth_lambda,
            "formula": FORMULA,
        }
    out = Path(ns.evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main(argv: list[str] | None = None) -> int:
    try:
        run(list(sys.argv[1:] if argv is None else argv))
    except AdapterError as e:
        print(f"minegs advanced trainer: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
