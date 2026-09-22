"""Running SfM over a frame set and recording what came out (§Phase 3 C1).

The backend composes and executes commands. This module decides everything around them, and
each decision is one the audit found being made silently:

* the images are the frame set's ``images/``, after ``check_frameset`` has re-read them, so a
  rejected frame cannot get in and an edited one cannot pass as the original;
* the rig comes from the crop records rather than a directory listing;
* the masks come from the frame set, with every image-to-mask binding already checked, because
  COLMAP ignores a mask it cannot match and says nothing;
* whichever reconstructions COLMAP produced are enumerated, and more than one is a refusal
  rather than an unannounced choice of the first.

The executor is a seam for the structural gate, which has no COLMAP. It is a Python argument.
There is no flag for it, because a reconstruction nobody reconstructed must not be something an
operator can produce by accident.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minegs.core.errors import ContractError
from minegs.core.provenance import make_id, stamp, tool_versions
from minegs.ingest.video.models import check_frameset, load_frameset
from minegs.ingest.video.sfm.base import SfMOptions, SfMRun, get_sfm_backend
from minegs.ingest.video.sfm.models import (
    SFM_FILE,
    SPARSE_FILES,
    CameraSummary,
    ComponentSummary,
    SfmRecord,
    check_sfm,
    model_digest,
)

#: (images_dir, work_dir, options) -> what ran. A substitute writes a model into
#: ``work_dir/sparse/0`` and returns a run with no commands, because none were executed.
SfmExecutor = Callable[[Path, Path, SfMOptions], SfMRun]


def enumerate_components(sparse_root: Path) -> list[Path]:
    """Every reconstruction under ``sparse/``, in COLMAP's own numbering order."""
    if not sparse_root.is_dir():
        return []
    found = [
        d
        for d in sorted(sparse_root.iterdir(), key=lambda p: p.name)
        if d.is_dir() and all((d / n).is_file() for n in SPARSE_FILES)
    ]
    return found


def _summaries(components: list[Path]) -> list[ComponentSummary]:
    from minegs.ingest.common import colmap_io

    out = []
    for path in components:
        model = colmap_io.read_model(path)
        out.append(
            ComponentSummary(
                name=path.name,
                registered_images=len(model.images),
                points=len(model.points3D),
                sha256=model_digest(path),
            )
        )
    return out


def _cameras(model_dir: Path) -> list[CameraSummary]:
    from minegs.ingest.common import colmap_io

    model = colmap_io.read_model(model_dir)
    return [
        CameraSummary(
            camera_id=cam.id,
            model=cam.model,
            width=cam.width,
            height=cam.height,
            params=[float(v) for v in cam.params],
        )
        for cam in sorted(model.cameras.values(), key=lambda c: c.id)
    ]


def run_sfm(
    frameset_dir: str | Path,
    out_dir: str | Path,
    *,
    backend: str = "colmap",
    mapper: str = "global",
    options: SfMOptions | None = None,
    component: str | None = None,
    executor: SfmExecutor | None = None,
    overwrite: bool = False,
) -> tuple[SfmRecord, Path]:
    """Reconstruct the frame set and write the artifact that says what was reconstructed."""
    rec_fs, fs_dir = load_frameset(frameset_dir)
    check_frameset(rec_fs, fs_dir)

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise ContractError(
            f"{out} is not empty. A reconstruction writes a database and a model tree; mixing "
            "two runs there leaves a model nobody can attribute. Use a new directory."
        )
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    opts = options or SfMOptions()
    images_dir = rec_fs.image_root(fs_dir)

    if rec_fs.crops:
        from minegs.ingest.video.rig import write_rig_config_from_crops

        opts.rig_config = write_rig_config_from_crops(rec_fs.crops, out / "rig_config.json")
    if rec_fs.masks:
        mask_root = rec_fs.mask_root(fs_dir)
        opts.masks_dir = mask_root

    if executor is None:
        run = get_sfm_backend(backend, mapper).run(images_dir, out, opts)
        real, backend_name = True, f"{backend}_{mapper}"
        version = tool_versions().get("colmap")
    else:
        run = executor(images_dir, out, opts)
        real, backend_name, version = False, f"substituted_{backend}_{mapper}", None

    components = enumerate_components(run.sparse_root)
    if not components:
        raise ContractError(
            f"SfM produced no reconstruction under {run.sparse_root}. Nothing registered, so "
            "there is no model to record; the command log is beside it."
        )
    summaries = _summaries(components)
    if component is None:
        if len(components) > 1:
            names = [s.name for s in summaries]
            sizes = {s.name: s.registered_images for s in summaries}
            raise ContractError(
                f"SfM produced {len(components)} separate reconstructions ({names}, registered "
                f"images {sizes}). They are disconnected pieces of the survey in unrelated "
                "coordinate frames, and taking the first is a choice, not a default. Name the "
                "one to use explicitly."
            )
        chosen = components[0]
    else:
        match = [c for c in components if c.name == component]
        if not match:
            raise ContractError(
                f"no reconstruction named {component!r} under {run.sparse_root}; COLMAP "
                f"produced {[c.name for c in components]}"
            )
        chosen = match[0]

    summary = next(s for s in summaries if s.name == chosen.name)
    model_rel = chosen.relative_to(out).as_posix()
    options_dump: dict[str, Any] = {
        "fix_intrinsics": opts.fix_intrinsics,
        "camera_model": opts.camera_model,
        "camera_params": opts.camera_params,
        "matcher": opts.matcher,
        "use_gpu": opts.use_gpu,
        "rig_config": opts.rig_config.name if opts.rig_config else None,
        "masks": bool(opts.masks_dir),
        "vocab_tree": opts.vocab_tree.name if opts.vocab_tree else None,
        "extra": dict(opts.extra),
    }
    record = SfmRecord(
        sfm_id=make_id("sfm"),
        frameset_id=rec_fs.frameset_id,
        images_sha256=rec_fs.images_sha256,
        backend=backend_name,
        backend_version=version,
        commands=run.commands,
        options=options_dump,
        intrinsics_source=opts.intrinsics_source,
        model_dir=model_rel,
        model_sha256=summary.sha256,
        components=summaries,
        selected_component=chosen.name,
        registered_images=summary.registered_images,
        points=summary.points,
        cameras=_cameras(chosen),
        real_sfm_execution=real,
        provenance=stamp(options_dump, parents=[rec_fs.frameset_id]),
    )
    check_sfm(record, out, images_sha256=rec_fs.images_sha256)
    (out / SFM_FILE).write_text(record.model_dump_json(indent=2))
    return record, out


__all__ = ["SfmExecutor", "enumerate_components", "run_sfm"]
