"""Depth maps -> LOCAL_METRIC surface samples (§1.7).

Two halves, deliberately split by phase:

* **Phase 1A (here).** Given depth maps that already exist — rendered elsewhere, or synthetic —
  plus the dataset's camera poses, produce a *surface artifact*: the points, and a record of
  where they came from. Pure numpy, no GPU, no backend.
* **Phase 1B.** ``render_depths`` — ask a trained run's backend for those depth maps. That needs
  the gsplat rasteriser and a GPU, and is not implemented.

The split matters twice over. It lets the artifact contract land without waiting on a
rasteriser integration — and it is the reason a Phase 1A surface cannot carry a geometry
accuracy claim: depth maps this code was handed are not evidence about the run they are
attributed to (``DepthSource`` in ``models.py``).
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.core.pointcloud import PointCloud
from minegs.eval.surface.models import (
    SURFACE_FILE,
    SURFACE_POINTS_FILE,
    SurfaceRecord,
)

DEPTH_SUFFIX = ".npy"


def backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    world_from_cam: SE3,
    stride: int = 1,
    max_depth: float | None = None,
) -> np.ndarray:
    """Depth (H,W, metres along z) -> world points (N,3)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    m = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        m &= z <= max_depth
    x = (u[m] + 0.5 - K[0, 2]) / K[0, 0] * z[m]
    y = (v[m] + 0.5 - K[1, 2]) / K[1, 1] * z[m]
    return world_from_cam.apply(np.column_stack([x, y, z[m]]))


def depth_map_path(depth_dir: str | Path, image_name: str) -> Path:
    """``<depth_dir>/<image stem>.npy`` — the naming contract (§11)."""
    return Path(depth_dir) / (Path(image_name).stem + DEPTH_SUFFIX)


def missing_depth_maps(depth_dir: str | Path, images: dict) -> list[str]:
    """Image names in *images* with no depth map in *depth_dir*."""
    return sorted(
        im.name for im in images.values() if not depth_map_path(depth_dir, im.name).is_file()
    )


def require_unique_stems(images: dict) -> None:
    """The depth naming contract is per *stem*, so two views sharing one is unresolvable.

    ``a/view.jpg`` and ``b/view.png`` both want ``view.npy``: on the write side the second
    render silently overwrites the first, on the read side both views back-project the same
    map. Rare, and worth a sentence rather than a silent wrong surface.
    """
    seen: dict[str, str] = {}
    for im in images.values():
        stem = Path(im.name).stem
        if stem in seen:
            raise ContractError(
                f"images {seen[stem]!r} and {im.name!r} share the depth-map name "
                f"{stem}{DEPTH_SUFFIX}; the depth contract is one map per image stem"
            )
        seen[stem] = im.name


def unexpected_depth_maps(depth_dir: str | Path, images: dict) -> list[str]:
    """Depth maps in *depth_dir* that match no camera view.

    The mirror of a missing map, and the louder symptom: a directory holding maps for views
    this dataset does not have is usually a directory belonging to another run or another
    dataset, which is exactly the mix-up the run checks exist to catch.
    """
    wanted = {depth_map_path(depth_dir, im.name).name for im in images.values()}
    return sorted(p.name for p in Path(depth_dir).glob(f"*{DEPTH_SUFFIX}") if p.name not in wanted)


def check_depth_shape(depth: np.ndarray, camera, name: str) -> None:
    """A depth map must be the resolution its intrinsics describe.

    Nothing downstream notices if it is not. ``backproject_depth`` walks whatever grid it is
    given and applies ``fx, fy, cx, cy`` from the camera, so a half-resolution map is
    back-projected with full-resolution intrinsics: every ray comes out at the wrong angle, the
    surface is quietly wrong, and the artifact, the record and the evaluation all succeed. Any
    rescaling has to adjust K, so this refuses rather than guessing which of the two is right.
    """
    expected = (camera.height, camera.width)
    if depth.shape != expected:
        raise ContractError(
            f"{name}: depth map is {depth.shape[1]}x{depth.shape[0]} but camera "
            f"{camera.id} is {camera.width}x{camera.height}. Back-projecting one resolution "
            "with another's intrinsics silently bends every ray; re-render at the camera's "
            "resolution, or add a camera whose intrinsics match the depth."
        )


def depth_digest(depth_dir: str | Path, images: dict) -> str:
    """Order-independent sha256 over the depth maps *images* consumes."""
    from minegs.core.provenance import sha256_file

    h = hashlib.sha256()
    for name in sorted(im.name for im in images.values()):
        f = depth_map_path(depth_dir, name)
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def depth_to_points(
    depth_dir: str | Path,
    cameras: dict,
    images: dict,
    stride: int = 2,
    max_depth: float | None = None,
    require_all: bool = False,
) -> PointCloud:
    """Fuse ``<depth_dir>/<image stem>.npy`` maps (rendered by the backend) into one cloud.

    ``require_all`` is what the production path passes. A view whose depth map is absent leaves
    a hole in the fused cloud, and a hole is indistinguishable from unreconstructed geometry
    once it reaches completeness, sections or volume — so an artifact built for evaluation
    refuses rather than quietly covering less of the tunnel than it claims (§12).
    """
    depth_dir = Path(depth_dir)
    if require_all:
        missing = missing_depth_maps(depth_dir, images)
        extra = unexpected_depth_maps(depth_dir, images)
        if missing or extra:
            raise ContractError(
                f"{depth_dir}: expected {len(images)} depth maps, found "
                f"{len(images) - len(missing)} of them"
                + (
                    f"; missing {missing[:6]}{' ...' if len(missing) > 6 else ''}"
                    if missing
                    else ""
                )
                + (
                    f"; {len(extra)} map(s) match no camera view {extra[:6]}"
                    f"{' ...' if len(extra) > 6 else ''}"
                    if extra
                    else ""
                )
                + ". Every camera view needs exactly one: a skipped view is a hole in the "
                "surface that reads as missing geometry downstream."
            )
    pts = []
    for im in images.values():
        f = depth_map_path(depth_dir, im.name)
        if not f.exists():
            continue
        depth = np.load(f)
        cam = cameras[im.camera_id]
        check_depth_shape(depth, cam, f.name)
        pts.append(backproject_depth(depth, cam.K(), im.world_from_cam, stride, max_depth))
    if not pts:
        raise FileNotFoundError(f"no depth maps in {depth_dir}")
    return PointCloud(np.concatenate(pts), frame="LOCAL_METRIC")


# ---------------------------------------------------------------- production surface artifact


@contextmanager
def staged_dir(out: Path) -> Iterator[Path]:
    """Write into a sibling ``.<name>.minegs-partial`` and rename on success (§15).

    Same shape as the ingest and dataset publishes: a surface directory either exists complete
    or does not exist, so an interrupted fusion cannot be picked up as a thinner surface.
    """
    tmp = out.parent / f".{out.name}.minegs-partial"
    if tmp.exists():
        raise ContractError(f"{tmp} exists from an interrupted run; inspect and remove it")
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(out)


def check_run(run_dir: Path, dataset_id: str, dataset_hash: str):
    """The §10 run checks. Returns the ``RunRecord``, so callers verify against it rather than
    against a bare id — checkpoint identity and render settings both live there."""
    from minegs.train.runner.base import RunRecord, RunStatus

    rpath = run_dir / "run.json"
    if not rpath.is_file():
        raise ContractError(f"{run_dir}: no run.json; --run-dir must point at a finished run")
    rec = RunRecord.load(rpath)
    if rec.status != RunStatus.SUCCEEDED:
        raise ContractError(
            f"run {rec.run_id} is {rec.status.value}, not succeeded; a surface may only be built "
            "from a run whose outputs were verified"
        )
    if rec.dataset_id != dataset_id:
        raise ContractError(
            f"run {rec.run_id} trained on dataset {rec.dataset_id!r}, not {dataset_id!r}"
        )
    if rec.dataset_hash != dataset_hash:
        raise ContractError(
            f"run {rec.run_id} trained on dataset_hash {rec.dataset_hash[:12]}, but "
            f"{dataset_id} now hashes to {dataset_hash[:12]}; the dataset changed after the run"
        )
    if rec.frame_of_outputs != "LOCAL_METRIC":
        raise ContractError(
            f"run {rec.run_id} declares frame_of_outputs={rec.frame_of_outputs}, not LOCAL_METRIC"
        )
    return rec


def _depth_provenance(
    depth_dir: Path, run, run_dir: Path, dataset_id: str, dataset_hash: str, model
) -> tuple[str, Any]:
    """``(depth_source, DepthManifest | None)`` for the maps in *depth_dir* (§1B §5)."""
    from minegs.eval.surface.models import (
        DepthManifest,
        find_depth_manifest,
        verify_depth_manifest,
    )

    found = find_depth_manifest(depth_dir)
    if found is None:
        return "external_unverified", None
    from minegs.eval.surface.render import require_metric_outputs, require_reproducible_render

    rendered = DepthManifest.load(found)
    verify_depth_manifest(
        rendered, depth_dir, run, run_dir, dataset_id, dataset_hash, model.cameras, model.images
    )
    # Re-asserted here rather than trusted from the manifest. The renderer checked both too,
    # but the manifest records no verdict, so depth written by an older build — or by one whose
    # guard was weaker — would otherwise promote on the strength of having a manifest at all.
    require_metric_outputs(run)
    require_reproducible_render(run, run_dir)
    return "minegs_render", rendered


def build_depth_surface(
    depth_dir: str | Path,
    dataset_dir: str | Path,
    run_dir: str | Path,
    out_dir: str | Path,
    stride: int = 2,
    max_depth: float | None = None,
) -> tuple[SurfaceRecord, Path]:
    """Fuse depth maps into a published surface artifact. Returns the record and its directory.

    The run checks establish that the *dataset* and the *run* belong together. Whether the
    depth maps belong to that run is a separate question, and the answer decides what the
    surface may claim:

    * a directory of bare ``.npy`` files cannot answer it — nothing in it ties the maps to the
      run — so the record is stamped ``external_unverified`` and the artifact is
      diagnostic-only;
    * a directory carrying a ``depth_manifest.json`` that still verifies against this run, this
      dataset and these bytes was written by ``minegs eval render-depth`` in the same pass that
      produced the maps, and earns ``minegs_render``.

    A manifest is never taken on its presence. It is checked, and a failed check is a refusal,
    not a demotion: a manifest that no longer agrees with its inputs is a sign something moved,
    which is exactly when quietly continuing would be worst.
    """
    from minegs.core.manifest import Manifest
    from minegs.core.pointcloud import write_ply
    from minegs.core.provenance import make_id, sha256_file, sha256_tree, stamp
    from minegs.ingest.common.colmap_io import read_model
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    depth_dir, dataset_dir = Path(depth_dir), Path(dataset_dir)
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if stride < 1:
        raise ContractError(f"--stride must be >= 1, got {stride}")
    if max_depth is not None and max_depth <= 0:
        raise ContractError(f"--max-depth must be > 0, got {max_depth}")
    if not depth_dir.is_dir():
        raise ContractError(f"{depth_dir}: no such depth directory")
    if out_dir.exists():
        raise ContractError(
            f"{out_dir} exists; a surface artifact is written once. Pass --out to name a new one."
        )

    manifest = Manifest.load_dataset(dataset_dir)  # strict layout: images/, sparse/0, init points
    model = read_model(dataset_dir / "sparse" / "0")
    if not model.images:
        raise ContractError(f"{dataset_dir}/sparse/0: no camera poses to back-project against")
    unknown = sorted({im.camera_id for im in model.images.values()} - set(model.cameras))
    if unknown:
        raise ContractError(f"{dataset_dir}/sparse/0: images reference unknown cameras {unknown}")
    require_unique_stems(model.images)
    dataset_hash = sha256_tree(dataset_dir, DATASET_HASH_PATTERNS)
    run = check_run(run_dir, manifest.dataset_id, dataset_hash)
    run_id = run.run_id
    depth_source, rendered = _depth_provenance(
        depth_dir, run, run_dir, manifest.dataset_id, dataset_hash, model
    )

    pc = depth_to_points(depth_dir, model.cameras, model.images, stride, max_depth, True)
    # §14 sanity. Neither is reachable from finite depth maps and a rigid pose, which is the
    # point: if one ever fires, the inputs were not what the contract says they are.
    if len(pc) == 0:
        raise ContractError(
            f"{depth_dir}: every depth sample was filtered out (non-finite, <= 0, or beyond "
            f"--max-depth); no surface to publish"
        )
    if not np.isfinite(pc.xyz).all():
        raise ContractError(f"{depth_dir}: back-projected points are not all finite")
    if pc.frame != "LOCAL_METRIC":
        raise ContractError(f"fused cloud is in frame {pc.frame}, expected LOCAL_METRIC")

    lo, hi = pc.xyz.min(axis=0), pc.xyz.max(axis=0)
    parameters = {
        "depth_dir": str(depth_dir),
        "stride": int(stride),
        "max_depth_m": None if max_depth is None else float(max_depth),
        "expected_views": len(model.images),
    }
    if rendered is not None:
        parameters["depth_manifest_id"] = rendered.manifest_id
        parameters["renderer"] = rendered.renderer.get("name")
        parameters["checkpoint"] = rendered.checkpoint.get("file")
    with staged_dir(out_dir) as tmp:
        ply = write_ply(pc, tmp / SURFACE_POINTS_FILE)
        rec = SurfaceRecord(
            surface_id=make_id("surface"),
            dataset_id=manifest.dataset_id,
            dataset_hash=dataset_hash,
            run_id=run_id,
            method="depth_backprojection",
            # Derived from the evidence on disk, never passed in by a caller: the only way to
            # reach the claim-capable value is to have rendered the depth with minegs.
            depth_source=depth_source,
            point_file=SURFACE_POINTS_FILE,
            point_sha256=sha256_file(ply),
            point_count=len(pc),
            depth_map_count=len(model.images),
            depth_sha256=depth_digest(depth_dir, model.images),
            parameters=parameters,
            bounds_min_m=[float(v) for v in lo],
            bounds_max_m=[float(v) for v in hi],
            span_m=[float(v) for v in (hi - lo)],
            provenance=stamp(parameters, parents=[manifest.dataset_id, run_id]),
        )
        rec.save(tmp / SURFACE_FILE)
    return rec, out_dir
