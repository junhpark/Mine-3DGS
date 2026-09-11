"""Stage a dataset for a backend (Phase 0D).

Why a staged copy instead of pointing the trainer at ``dataset/``:

1. **Read-only input.** The dataset mount is ``:ro`` in docker and gsplat's COLMAP parser
   writes downscaled image folders (``images_<factor>_png``) next to ``images/``.
2. **Profile image subset.** ``profile.max_images`` (light: 100) is applied *here*, evenly
   spaced over the manifest's train images, so the trainer never sees test-group images.
3. **TLS initialisation.** ``init_points.ply`` (LOCAL_METRIC) becomes ``sparse/0/points3D.txt``
   so ``--init_type sfm`` initialises from TLS points, not the SfM sparse cloud.

The staged directory lives under ``runs/<run_id>/staged`` and its hash is recorded in
``run.json`` next to the full dataset hash (§9).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.ingest.common import colmap_io

MAX_INIT_POINTS = 4_000_000


@dataclass
class StagedDataset:
    path: Path
    images: list[str]
    n_train_available: int
    subset: bool
    init_points: int
    init_source: str  # "init_points.ply" | "points3D.txt"


def select_images(train_images: list[str], max_images: int | None) -> list[str]:
    """Evenly spaced subset preserving order; all images when max_images is None/large."""
    if max_images is None or max_images >= len(train_images):
        return list(train_images)
    if max_images < 1:
        raise ContractError("max_images must be >= 1")
    idx = np.unique(np.linspace(0, len(train_images) - 1, max_images).round().astype(int))
    return [train_images[i] for i in idx]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # same filesystem: no extra space
    except OSError:
        shutil.copy2(src, dst)


def stage_dataset(
    dataset_dir: Path,
    staged_dir: Path,
    manifest: Manifest | None = None,
    max_images: int | None = None,
    use_init_points: bool = True,
    chunk_id: str | None = None,
) -> StagedDataset:
    dataset_dir = Path(dataset_dir)
    staged_dir = Path(staged_dir)
    manifest = manifest or Manifest.load_dataset(dataset_dir)
    model = colmap_io.read_model(dataset_dir / "sparse" / "0")
    by_name = model.image_by_name()

    train = manifest.train_images()
    if chunk_id:
        if manifest.chunks is None:
            raise ContractError("chunk_id given but manifest has no chunks")
        chunk = next((c for c in manifest.chunks.items if c.id == chunk_id), None)
        if chunk is None:
            raise ContractError(f"unknown chunk {chunk_id}")
        allowed = set(manifest.images_of(chunk.groups))
        train = [n for n in train if n in allowed]
    missing = [n for n in train if n not in by_name]
    if missing:
        raise ContractError(
            f"{len(missing)} train images missing from sparse/0 (e.g. {missing[:3]})"
        )
    chosen = select_images(train, max_images)
    if not chosen:
        raise ContractError("no train images to stage")

    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    (staged_dir / "images").mkdir(parents=True)
    keep_ids = set()
    for name in chosen:
        _link_or_copy(dataset_dir / "images" / name, staged_dir / "images" / name)
        keep_ids.add(by_name[name].id)
    if (dataset_dir / "masks").is_dir():
        for name in chosen:
            m = dataset_dir / "masks" / (name + ".png")
            if m.exists():
                _link_or_copy(m, staged_dir / "masks" / (name + ".png"))

    images = {iid: im for iid, im in model.images.items() if iid in keep_ids}
    init_src = "points3D.txt"
    points = model.points3D
    if use_init_points:
        pc = read_ply(dataset_dir / manifest.initialization.file)
        if pc.frame not in ("LOCAL_METRIC", "UNKNOWN"):
            raise ContractError(f"init_points.ply must be LOCAL_METRIC, got {pc.frame}")
        pc = pc.subsample(MAX_INIT_POINTS)
        rgb = pc.rgb if pc.rgb is not None else np.full((len(pc), 3), 128, np.uint8)
        points = {i + 1: colmap_io.Point3D(i + 1, pc.xyz[i], rgb[i]) for i in range(len(pc))}
        init_src = manifest.initialization.file
        # tracks would reference the old points: clear them
        for im in images.values():
            im.xys = np.zeros((0, 2))
            im.point3D_ids = np.zeros(0, dtype=np.int64)
    cams = {
        cid: c
        for cid, c in model.cameras.items()
        if any(im.camera_id == cid for im in images.values())
    }
    staged_model = colmap_io.ColmapModel(cams, images, points)
    colmap_io.write_model(staged_model, staged_dir / "sparse" / "0")
    (staged_dir / "STAGED_FROM.txt").write_text(
        f"dataset_id={manifest.dataset_id}\nsource={dataset_dir}\nimages={len(chosen)}/{len(train)}\ninit={init_src}\n"
    )
    return StagedDataset(
        path=staged_dir,
        images=chosen,
        n_train_available=len(train),
        subset=len(chosen) < len(train),
        init_points=len(points),
        init_source=init_src,
    )
