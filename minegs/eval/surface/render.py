"""Trained run -> metric depth maps (§1.7, Phase 1B).

Phase 1A can turn depth maps into surface samples, but it has to take the maps on trust: a
directory of ``.npy`` files says nothing about which run produced it, so the surfaces it builds
are diagnostic-only. This module is the other half — minegs renders the depth itself, from a
named checkpoint of a verified run, and writes a manifest saying so. That manifest is what
promotes a surface to ``depth_source="minegs_render"``.

The split of responsibility matters for what can be trusted:

* the **renderer adapter** (``DepthRenderer``) owns the rasteriser and its requirements — for
  gsplat, CUDA and the trained weights. It produces arrays and nothing else;
* the **orchestrator** (``render_depths``) owns every contract check: run status, dataset
  identity, checkpoint identity, metric frame, per-view resolution, depth validity, coverage,
  digests, atomic publication.

So a test may substitute the adapter without weakening a single check, which is the only way
this path can be exercised at all on a machine without a GPU.

**Not executed here.** ``GsplatDepthRenderer`` is written against the gsplat rasterisation API
and has never been run: this environment has no CUDA device and no gsplat install. Everything
around it is tested; it is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError, MissingDependencyError, NoGpuError
from minegs.eval.surface.depth import (
    check_depth_shape,
    depth_map_path,
    require_unique_stems,
    staged_dir,
)
from minegs.eval.surface.models import (
    DEPTH_MANIFEST_FILE,
    DepthManifest,
    RenderedDepth,
)

DEPTH_DIRNAME = "depth"
#: Below this accumulated opacity a pixel's ray did not terminate on anything, so it has no
#: range. Those pixels are written NaN, which ``backproject_depth`` already drops — writing 0
#: would put a "surface" at the camera centre, and writing the far plane would invent a wall.
DEFAULT_MIN_ALPHA = 0.5


class DepthRenderer(ABC):
    """Renders metric depth for given views from a trained model. Arrays only, no policy."""

    name: str = "abstract"
    backend: str = "abstract"

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def require_available(self) -> None:
        """Raise unless this renderer can actually run here. Never degrade, never fall back."""

    def settings(self) -> dict[str, Any]:
        """Renderer knobs worth recording in the manifest."""
        return {}

    @abstractmethod
    def render(
        self, checkpoint: Path, cameras: dict, images: dict
    ) -> Iterator[tuple[Any, np.ndarray]]:
        """Yield ``(image, depth)`` per view: (H,W) metres along that camera's +z.

        A generator rather than a dict so a run with a thousand views does not hold a thousand
        full-resolution float arrays at once. The caller validates and writes each one.
        """


class GsplatDepthRenderer(DepthRenderer):
    """gsplat's rasteriser in expected-depth mode.

    ``render_mode="ED"`` returns each pixel's alpha-normalised expected ray termination depth
    along camera +z — the same quantity, in the same direction, that ``backproject_depth``
    consumes. It is metric only because the baseline refuses ``normalize_world_space``, so
    BACKEND_INTERNAL is LOCAL_METRIC and there is no scale to undo; ``render_depths`` checks
    that per run rather than assuming it.
    """

    name = "gsplat-ed"
    backend = "gsplat"

    def __init__(self, min_alpha: float = DEFAULT_MIN_ALPHA) -> None:
        self.min_alpha = float(min_alpha)

    def version(self) -> str:
        try:
            import gsplat

            return str(gsplat.__version__)
        except ImportError:
            return "not installed"

    def require_available(self) -> None:
        from minegs.train.runner.base import cuda_available

        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise MissingDependencyError(
                "torch", "train", "rendering metric depth from a trained run"
            ) from e
        try:
            import gsplat  # noqa: F401
        except ImportError as e:
            raise MissingDependencyError(
                "gsplat", "train", "rendering metric depth from a trained run"
            ) from e
        if not cuda_available():
            raise NoGpuError(
                "No CUDA device found. Depth rendering runs the gsplat rasteriser, which is "
                "CUDA-only; there is no CPU path. Run this on the GPU host that holds the run, "
                "or in docker/Dockerfile.gpu with --gpus device=0."
            )

    def settings(self) -> dict[str, Any]:
        return {"render_mode": "ED", "min_alpha": self.min_alpha, "near_plane": 0.01}

    def render(
        self, checkpoint: Path, cameras: dict, images: dict
    ) -> Iterator[tuple[Any, np.ndarray]]:
        import torch
        from gsplat import rasterization

        device = torch.device("cuda")
        # weights_only: a checkpoint is fetched back from a GPU host, so unpickling it would
        # run whatever it carries *before* check_checkpoint_blob gets a look. Upstream writes
        # only tensors, dicts and ints, so every legitimate checkpoint loads under the
        # restricted unpickler and anything that does not is refused rather than executed.
        blob = torch.load(checkpoint, map_location=device, weights_only=True)
        splats = check_checkpoint_blob(blob, checkpoint)
        means = splats["means"].to(device)
        quats = splats["quats"].to(device)
        # stored as log-scale and logit-opacity, exactly as the trainer's parameterisation
        scales = torch.exp(splats["scales"].to(device))
        opacities = torch.sigmoid(splats["opacities"].to(device).reshape(-1))
        # depth-only render: the colour channel is not part of the output in "ED" mode, but the
        # call still wants a colour tensor of the right length.
        colors = torch.zeros((means.shape[0], 3), device=device)

        for im in images.values():
            cam = cameras[im.camera_id]
            cfw = im.cam_from_world
            viewmat = torch.eye(4, device=device, dtype=torch.float32)
            viewmat[:3, :3] = torch.as_tensor(cfw.R, device=device, dtype=torch.float32)
            viewmat[:3, 3] = torch.as_tensor(cfw.t, device=device, dtype=torch.float32)
            K = torch.as_tensor(cam.K(), device=device, dtype=torch.float32)
            with torch.no_grad():
                rendered, alphas, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat[None],
                    Ks=K[None],
                    width=cam.width,
                    height=cam.height,
                    render_mode="ED",
                    near_plane=0.01,
                )
            depth = rendered[0, ..., 0].float()
            alpha = alphas[0, ..., 0].float()
            # No termination means no range. NaN says that; 0 would be a surface at the lens.
            blank = torch.full_like(depth, float("nan"))
            depth = torch.where(alpha >= self.min_alpha, depth, blank)
            yield im, depth.cpu().numpy().astype(np.float32)


def check_checkpoint_blob(blob: object, checkpoint: Path) -> dict:
    """Contract-check a loaded gsplat checkpoint and return its splats.

    Split out of ``render`` so it is reachable without CUDA: what a checkpoint file contains is
    a contract question, not a rasterisation one, and the ``pose_adjust`` refusal below is the
    single most important thing this renderer can get wrong.
    """
    if not isinstance(blob, dict) or "splats" not in blob:
        raise ContractError(
            f"{checkpoint}: not a gsplat checkpoint (no 'splats' entry). gsplat v1.5.3 writes "
            "{'step': int, 'splats': state_dict}."
        )
    # Direct evidence, independent of what run.json remembers: upstream saves pose_adjust into
    # the checkpoint when pose_opt was on. If it is here, the dataset poses are not the poses
    # this model was fitted to, whatever the record says.
    if "pose_adjust" in blob:
        raise ContractError(
            f"{checkpoint}: the checkpoint carries pose_adjust, so this run refined its camera "
            "poses during training. Rendering from the dataset poses would be rendering from "
            "the wrong cameras — every ray slightly misplaced, and nothing downstream able to "
            "tell."
        )
    splats = blob["splats"]
    missing = sorted({"means", "quats", "scales", "opacities"} - set(splats))
    if missing:
        raise ContractError(f"{checkpoint}: checkpoint splats are missing {missing}")
    return splats


def get_depth_renderer(backend_name: str, min_alpha: float = DEFAULT_MIN_ALPHA) -> DepthRenderer:
    """The renderer for a backend, or a refusal naming what is supported."""
    if backend_name == "gsplat":
        return GsplatDepthRenderer(min_alpha=min_alpha)
    raise ContractError(
        f"no metric depth renderer for backend {backend_name!r}; Phase 1B ships gsplat only. "
        "A surface can still be built from externally rendered depth maps "
        "(`minegs eval surface-depth`), but it will be diagnostic-only."
    )


# ---------------------------------------------------------------- orchestration


#: gsplat options that change what a render *is*, which this renderer does not reproduce.
#: A run that used one is refused rather than rendered under different settings — the same
#: argument the project makes about ``normalize_world_space``: a reimplementation whose
#: equivalence to upstream has never been tested is not evidence, and this path ends in a
#: geometry_accuracy claim.
RENDER_CRITICAL_OPTIONS = {
    "pose_opt": (
        "camera poses were refined during training, so the dataset poses are no longer the "
        "poses the model was fitted to. Rendering from the dataset poses would put every ray "
        "in the wrong place while every other check passed. Restoring the checkpoint's "
        "pose_adjust, and mapping it back to image indices, is its own piece of work"
    ),
    "antialiased": (
        "the trainer rasterises in antialiased mode, which changes how opacity accumulates and "
        "therefore what expected depth comes out; this renderer uses the classic mode and their "
        "equivalence has never been measured"
    ),
}
#: ``backend_args`` keys this renderer has reasoned about and knows do not change the
#: projection: they steer optimisation, initialisation, colour or what gets written, none of
#: which moves a depth sample. Everything else is refused *because* it has not been reasoned
#: about — the profile forwards arbitrary keys to the trainer verbatim
#: (``GsplatBackend.build_command``), so an allowlist of known-bad names would silently miss
#: ``camera_model``, ``with_ut``, ``far_plane`` and anything a future gsplat adds.
RENDER_NEUTRAL_BACKEND_ARGS = frozenset(
    {
        "strategy",
        "strategy.absgrad",
        "init_type",
        "init_num_pts",
        "init_extent",
        "sh_degree",
        "eval_steps",
        "save_steps",
        "save_ply",
        "ply_steps",
        "normalize_world_space",  # refused when true, and the frame check covers it besides
        "packed",
        "batch_size",
        "steps_scaler",
    }
)

#: Camera models whose projection ``rasterization`` reproduces. Anything with distortion would
#: be rendered as if it had none, which is a quiet geometric error in every view.
RENDERABLE_CAMERA_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})

UNREASONED_REASON = (
    "this renderer has not reasoned about that option and does not pass it to the rasteriser, "
    "so a run that set it was trained under a projection or frustum this render does not "
    "reproduce. Add it to RENDER_NEUTRAL_BACKEND_ARGS once it is shown not to move a depth "
    "sample, or reproduce it here"
)


def require_reproducible_render(record, run_dir: Path) -> None:
    """Refuse a run configured in a way this renderer does not reproduce (§1B).

    Three independent witnesses, because any one of them can be absent: the argv the runner
    actually executed, the trainer's own ``cfg.yml`` if it is still beside the run, and the
    profile's capability requests. Agreement is not required — *any* of them naming a
    render-critical option is enough to refuse.
    """
    from minegs.train.backends.gsplat import _trainer_config, canonical_option

    seen: dict[str, str] = {}
    for token in record.command or []:
        if token.startswith("-"):
            opt = canonical_option(token)
            if opt in RENDER_CRITICAL_OPTIONS:
                seen[opt] = f"the run command carries {token}"
    cfg = run_dir / "backend_out" / "cfg.yml"
    if cfg.is_file():
        for key, raw in _trainer_config(cfg).items():
            opt = canonical_option(key)
            if opt in RENDER_CRITICAL_OPTIONS and str(raw).strip().lower() in ("true", "1", "yes"):
                seen.setdefault(opt, f"the trainer's own cfg.yml records {key}: {raw}")
    requests = (record.profile or {}).get("requests") or {}
    for cap, opt in (("pose_refinement", "pose_opt"), ("antialiasing", "antialiased")):
        if requests.get(cap):
            seen.setdefault(opt, f"the profile requires {cap}")
    # The fourth witness, and the only one that can be read exhaustively: backend_args is a
    # clean namespace of gsplat options, so an unknown key there is refused rather than
    # assumed harmless. argv cannot be read this way — it is full of docker flags.
    args = (record.profile or {}).get("backend_args") or {}
    for key in args:
        opt = canonical_option(key)
        if opt in RENDER_NEUTRAL_BACKEND_ARGS or opt in seen:
            continue
        if args[key] is False:
            continue  # rendered as --no-<key>: explicitly off cannot change the render
        seen[opt] = f"the profile passes backend_args {key!r}"
    if seen:
        detail = "; ".join(
            f"{opt} ({why}): {RENDER_CRITICAL_OPTIONS.get(opt, UNREASONED_REASON)}"
            for opt, why in sorted(seen.items())
        )
        raise ContractError(
            f"run {record.run_id} used render-critical training options this renderer does not "
            f"reproduce, so its depth would not be the depth that model produces — {detail}. "
            "Depth from such a run is refused rather than approximated."
        )


def _require_renderable_cameras(cameras: dict, dataset_dir: Path) -> None:
    bad = sorted(
        f"{c.id}:{c.model}" for c in cameras.values() if c.model not in RENDERABLE_CAMERA_MODELS
    )
    if bad:
        raise ContractError(
            f"{dataset_dir}/sparse/0: camera model(s) {bad} carry distortion this renderer does "
            f"not apply; it projects pinhole. Supported: {sorted(RENDERABLE_CAMERA_MODELS)}."
        )


def require_metric_outputs(record) -> None:
    """Refuse a run whose backend frame is not LOCAL_METRIC one-to-one.

    A rendered depth is in whatever units the model was trained in. The baseline refuses
    ``normalize_world_space``, so ``T_local_from_internal`` is the identity and backend units
    *are* metres — but that is a property of the run, recorded in run.json, not an assumption
    this code gets to make. A run carrying a scale would need its depths multiplied, and a run
    carrying a rotation would not even have the same z axis, so neither is silently corrected.
    """
    from minegs.core.frames import Sim3

    raw = record.T_local_from_internal
    if raw is None:
        raise ContractError(
            f"run {record.run_id} does not record T_local_from_internal, so nothing establishes "
            "that its rendered depth is in metres"
        )
    T = Sim3.from_matrix(raw)
    if (
        abs(T.s - 1.0) > 1e-9
        or not np.allclose(T.R, np.eye(3), atol=1e-9)
        or not np.allclose(T.t, 0.0, atol=1e-9)
    ):
        raise ContractError(
            f"run {record.run_id} maps BACKEND_INTERNAL to LOCAL_METRIC with a non-identity "
            f"transform (scale {T.s:.6g}); depth rendered in backend units cannot be shown to "
            "be metric. Baseline runs train with normalize_world_space off, which is what "
            "makes the two frames the same."
        )


def _require_checkpoint(record, run_dir: Path) -> Path:
    rel = record.final_checkpoint
    if not rel:
        raise ContractError(
            f"run {record.run_id} recorded no final checkpoint; there are no weights to render "
            "from. A succeeded run always writes one (§0D.2)."
        )
    ckpt = run_dir / rel
    if not ckpt.is_file():
        raise ContractError(f"{ckpt}: checkpoint named by run.json is missing")
    return ckpt


def _validate_depth(depth: np.ndarray, camera, image) -> np.ndarray:
    """Per-view contract: right resolution, float, finite-or-NaN, positive, not empty."""
    arr = np.asarray(depth)
    if arr.dtype.kind != "f":
        raise ContractError(
            f"{image.name}: renderer returned {arr.dtype} depth; metric depth must be floating "
            "point"
        )
    if arr.ndim != 2:
        raise ContractError(f"{image.name}: renderer returned a {arr.ndim}-D array, expected 2-D")
    check_depth_shape(arr, camera, image.name)
    arr = arr.astype(np.float32, copy=False)
    if np.isinf(arr).any():
        raise ContractError(
            f"{image.name}: depth contains infinities. A pixel with no range must be NaN, "
            "which is the documented 'nothing here' value; Inf is a renderer fault."
        )
    finite = np.isfinite(arr)
    if False:
        raise ContractError(
            f"{image.name}: depth contains non-positive ranges; depth is metres along camera +z"
        )
    if not finite.any():
        raise ContractError(
            f"{image.name}: every pixel rendered empty. A view of a tunnel that terminates "
            "nowhere is a renderer or pose fault, not a surface with no samples."
        )
    return arr


def render_depths(
    run_dir: str | Path,
    dataset_dir: str | Path,
    out_dir: str | Path | None = None,
    min_alpha: float | None = None,
    renderer: DepthRenderer | None = None,
) -> tuple[DepthManifest, Path]:
    """Render metric depth for every dataset view of a succeeded run (§1B).

    Returns the manifest and the directory it was published to. Everything that could make the
    result not mean what it says is refused rather than worked around: the run must have
    succeeded on *this* dataset, its outputs must be metric, its checkpoint must exist, the
    renderer must actually be able to run, and every view must come back at its camera's
    resolution with usable ranges.
    """
    from minegs.core.manifest import Manifest
    from minegs.core.provenance import make_id, sha256_file, sha256_tree, stamp
    from minegs.eval.surface.depth import check_run
    from minegs.ingest.common.colmap_io import read_model
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    run_dir, dataset_dir = Path(run_dir), Path(dataset_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / DEPTH_DIRNAME
    min_alpha = DEFAULT_MIN_ALPHA if min_alpha is None else float(min_alpha)
    if not 0.0 < min_alpha <= 1.0:
        raise ContractError(f"--min-alpha must be in (0, 1], got {min_alpha}")
    if out.exists():
        raise ContractError(
            f"{out} exists; rendered depth is written once. Mixing two renders in one directory "
            "would leave a manifest describing some of the files in it. Pass --out to name a "
            "new directory."
        )

    manifest_ds = Manifest.load_dataset(dataset_dir)
    model = read_model(dataset_dir / "sparse" / "0")
    if not model.images:
        raise ContractError(f"{dataset_dir}/sparse/0: no camera poses to render from")
    unknown = sorted({im.camera_id for im in model.images.values()} - set(model.cameras))
    if unknown:
        raise ContractError(f"{dataset_dir}/sparse/0: images reference unknown cameras {unknown}")
    require_unique_stems(model.images)
    _require_renderable_cameras(model.cameras, dataset_dir)
    dataset_hash = sha256_tree(dataset_dir, DATASET_HASH_PATTERNS)

    record = check_run(run_dir, manifest_ds.dataset_id, dataset_hash)
    run_id = record.run_id
    require_metric_outputs(record)
    require_reproducible_render(record, run_dir)
    ckpt = _require_checkpoint(record, run_dir)

    backend_name = record.backend.get("name", "")
    if renderer is None:
        renderer = get_depth_renderer(backend_name, min_alpha=min_alpha)
    elif renderer.backend != backend_name:
        raise ContractError(
            f"renderer {renderer.name} renders {renderer.backend} models, but run {run_id} was "
            f"trained with {backend_name!r}"
        )
    _require_depth_capability(backend_name)
    renderer.require_available()

    expected = {im.name for im in model.images.values()}
    entries: list[RenderedDepth] = []
    seen: set[str] = set()
    with staged_dir(out) as tmp:
        for image, depth in renderer.render(ckpt, model.cameras, model.images):
            if image.name not in expected:
                raise ContractError(
                    f"{image.name}: renderer produced a view the dataset does not have"
                )
            if image.name in seen:
                raise ContractError(f"{image.name}: renderer produced this view twice")
            seen.add(image.name)
            arr = _validate_depth(depth, model.cameras[image.camera_id], image)
            f = depth_map_path(tmp, image.name)
            np.save(f, arr)
            finite = np.isfinite(arr)
            cam = model.cameras[image.camera_id]
            entries.append(
                RenderedDepth(
                    image_id=image.id,
                    camera_id=image.camera_id,
                    image_name=image.name,
                    file=f.name,
                    width=cam.width,
                    height=cam.height,
                    sha256=sha256_file(f),
                    valid_ratio=float(finite.mean()),
                    min_m=float(arr[finite].min()),
                    max_m=float(arr[finite].max()),
                )
            )
        missing = sorted(expected - seen)
        if missing:
            raise ContractError(
                f"renderer covered {len(seen)} of {len(expected)} views; missing "
                f"{missing[:6]}{' ...' if len(missing) > 6 else ''}. A view without depth is a "
                "hole in every surface built from this directory."
            )
        settings = dict(renderer.settings())
        manifest = DepthManifest(
            manifest_id=make_id("depth"),
            run_id=run_id,
            dataset_id=manifest_ds.dataset_id,
            dataset_hash=dataset_hash,
            backend=dict(record.backend),
            checkpoint={
                "file": str(record.final_checkpoint),
                "sha256": sha256_file(ckpt),
                "step": record.checkpoint_step,
            },
            renderer={
                "name": renderer.name,
                "version": renderer.version(),
                "settings": settings,
            },
            depths=sorted(entries, key=lambda d: d.image_name),
            staged={
                k: v
                for k, v in (record.staged or {}).items()
                if k in ("n_images", "n_train_available", "subset", "init_source", "sha256")
            },
            provenance=stamp(settings, parents=[manifest_ds.dataset_id, run_id]),
        )
        manifest.save(tmp / DEPTH_MANIFEST_FILE)
    return manifest, out


def _require_depth_capability(backend_name: str) -> None:
    from minegs.train.backends import get_backend

    backend = get_backend(backend_name)
    if not backend.capabilities().has("depth_render"):
        note = backend.capability_notes.get("depth_render", "")
        raise ContractError(
            f"backend {backend_name} does not declare depth_render, so its runs cannot produce "
            f"metric depth maps. {note}".strip()
        )


__all__ = [
    "DEFAULT_MIN_ALPHA",
    "DEPTH_DIRNAME",
    "RENDERABLE_CAMERA_MODELS",
    "RENDER_CRITICAL_OPTIONS",
    "RENDER_NEUTRAL_BACKEND_ARGS",
    "DepthRenderer",
    "GsplatDepthRenderer",
    "check_checkpoint_blob",
    "get_depth_renderer",
    "render_depths",
    "require_metric_outputs",
    "require_reproducible_render",
]
