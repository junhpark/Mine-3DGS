"""Stage a dataset for a backend (Phase 0D).

Why a staged copy instead of pointing the trainer at ``dataset/``:

1. **Read-only input.** The dataset mount is ``:ro`` in docker and gsplat's COLMAP parser
   writes downscaled image folders (``images_<factor>_png``) next to ``images/``.

What the pinned upstream parser needs from the staged tree, found in the Phase 4 C0 audit
(docs/PHASE4_CONTRACT.md §2.3) and provided here:

* **a binary model.** The pycolmap fork gsplat v1.5.3 reads models with cannot parse COLMAP
  text on Python 3, so ``sparse/0`` carries ``*.bin`` beside the ``*.txt`` MineGS reads.
* **``images_<factor>/`` for ``data_factor > 1``.** ``Parser`` refuses to start without it
  (``colmap.py:183-187``). For ``.png`` images MineGS writes the downscaled copies itself, with
  upstream's own resize (PIL bicubic, ``round(w / f)``). For ``.jpg`` it links the originals,
  because upstream then resizes from ``images/`` into ``images_<factor>_png`` itself.
* **finite appearance colours.** With ``app_opt`` upstream initialises colour as
  ``logit(rgb / 255)``, so a 0 or 255 channel becomes an infinite parameter. When appearance
  is requested the staged init colours are clamped to ``[1, 254]``, and the clamp is recorded.
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
    #: How ``images_<factor>/`` was provided (None for factor 1).
    downscale: dict | None = None
    #: How many init colour channels were moved off 0/255 for ``app_opt`` (None: no clamp).
    init_rgb_clamped: int | None = None
    #: Phase 5: what a chunk stage selected (None for a run that is not a plan's chunk).
    chunk: dict | None = None


#: Every staged file the trainer reads, for the staged-tree hash in run.json.
STAGED_HASH_PATTERNS = (
    "sparse/0/*.txt",
    "sparse/0/*.bin",
    "images/**/*",
    "images_*/**/*",
    "masks/**/*",
    "supervision/**/*",
)
#: Upstream's own downscale (examples/datasets/colmap.py::_resize_image_folder).
DOWNSCALE_RULE = "PIL BICUBIC to (round(w / f), round(h / f)), RGB, PNG"
#: ``logit(rgb / 255)`` is finite only strictly inside (0, 255).
APPEARANCE_RGB_RANGE = (1, 254)


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


def _png_key(name: str) -> str:
    return os.path.splitext(name)[0] + ".png"


def stage_downscaled(staged_dir: Path, names: list[str], factor: int) -> dict | None:
    """Provide ``images_<factor>/`` the way the pinned upstream parser expects it."""
    if factor <= 1:
        return None
    exts = {os.path.splitext(n)[1] for n in names}
    lower = {e.lower() for e in exts}
    folder = staged_dir / f"images_{factor}"
    if lower == {".jpg"}:
        # Upstream sees a .jpg first entry and resizes images/ into images_<f>_png itself, then
        # pairs the two folders by sorted order. That pairing must be the identity.
        if sorted(_png_key(n) for n in names) != [_png_key(n) for n in sorted(names)]:
            raise ContractError(
                "upstream pairs images/ with its resized .png copies by sorted name, and for "
                "these file names the two orders differ; training would pair images with the "
                "wrong cameras. Rename the images, or train with data_factor 1."
            )
        for n in names:
            _link_or_copy(staged_dir / "images" / n, folder / n)
        return {
            "factor": factor,
            "folder": folder.name,
            "mode": "upstream_resize_from_jpg",
            "rule": DOWNSCALE_RULE,
        }
    if lower != {".png"} or len(exts) != 1:
        raise ContractError(
            f"data_factor {factor} with image suffixes {sorted(exts)}: upstream reads a "
            "downscaled folder as given for .png and resizes it itself for .jpg; for anything "
            "else (or a mix) which pixels it trains on is not defined. Use data_factor 1."
        )
    from PIL import Image as PILImage

    for n in names:
        with PILImage.open(staged_dir / "images" / n) as im:
            arr = np.asarray(im)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ContractError(
                f"{n} is not an RGB image (shape {arr.shape}); upstream slices [..., :3]"
            )
        h, w = arr.shape[:2]
        size = (round(w / factor), round(h / factor))
        out = folder / n
        out.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(np.ascontiguousarray(arr[..., :3])).resize(size, PILImage.BICUBIC).save(
            out
        )
    import PIL

    return {
        "factor": factor,
        "folder": folder.name,
        "mode": "minegs_png",
        "rule": DOWNSCALE_RULE,
        "pillow": PIL.__version__,
    }


def stage_depth_supervision(verified, staged_dir: Path) -> Path:
    """Copy a verified depth supervision artifact into the staged tree, byte for byte."""
    from minegs.core.provenance import sha256_tree

    dst = Path(staged_dir) / "supervision" / "depth" / verified.record.supervision_id
    if dst.exists():
        shutil.rmtree(dst)
    for f in sorted(p for p in Path(verified.path).iterdir() if p.is_file()):
        _link_or_copy(f, dst / f.name)
    got = sha256_tree(dst)
    if got != verified.artifact_sha256:
        raise ContractError(
            f"the staged copy of {verified.record.supervision_id} hashes to {got[:12]}, not the "
            f"verified {verified.artifact_sha256[:12]}"
        )
    return dst


def stage_dataset(
    dataset_dir: Path,
    staged_dir: Path,
    manifest: Manifest | None = None,
    max_images: int | None = None,
    use_init_points: bool = True,
    chunk_id: str | None = None,
    data_factor: int = 1,
    clamp_init_rgb: bool = False,
    chunk=None,
) -> StagedDataset:
    """Stage a dataset, or one chunk of it, for the pinned upstream trainer.

    ``chunk`` is a ``PlannedChunk`` of a verified ``ChunkPlanRecord`` (Phase 5): its images are the
    plan's selection, which must be exactly the global training images of its groups, and its init
    is the init points the existing support locator places inside the chunk's support. Poses and
    points are not re-expressed: a chunk lives in the dataset's LOCAL_METRIC frame.
    """
    dataset_dir = Path(dataset_dir)
    staged_dir = Path(staged_dir)
    manifest = manifest or Manifest.load_dataset(dataset_dir)
    model = colmap_io.read_model(dataset_dir / "sparse" / "0")
    by_name = model.image_by_name()

    if chunk_id is not None and chunk is None:
        raise ContractError(
            f"chunk {chunk_id!r}: a chunk is staged from a verified chunk plan (Phase 5 AD-1), "
            "not from the legacy manifest.chunks windows"
        )
    train = manifest.train_images()
    chunk_evidence: dict | None = None
    if chunk is not None:
        if not use_init_points:
            raise ContractError(
                "a chunk's init is selected from init_points.ply by chainage; the SfM sparse-"
                "track init path is not chunked"
            )
        global_train = set(train) - set(manifest.test_images())
        planned = set(chunk.images)
        if not planned <= global_train:
            raise ContractError(
                f"chunk {chunk.chunk_id} names {sorted(planned - global_train)[:3]} as training "
                "images, which the dataset's split does not train on"
            )
        want = {
            x for g in chunk.capture_groups for x in manifest.capture_groups[g].members
        } & global_train
        if want != planned:
            raise ContractError(
                f"chunk {chunk.chunk_id}: its images are not the training members of its groups "
                f"{chunk.capture_groups}; a group is staged whole or not at all"
            )
        from minegs.train.runner.base import require_whole_chunk

        require_whole_chunk(chunk, max_images)
        train = [n for n in train if n in planned]
        chunk_evidence = {"chunk_id": chunk.chunk_id, "images": sorted(planned)}
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
    if not use_init_points:
        # Tracks naming an image that was not staged make upstream's parser raise KeyError
        # (colmap.py:210); keep only the observations of staged images.
        points = {}
        for pid, p in model.points3D.items():
            keep = np.isin(np.asarray(p.image_ids), list(keep_ids))
            points[pid] = colmap_io.Point3D(
                pid,
                p.xyz,
                p.rgb,
                p.error,
                np.asarray(p.image_ids)[keep],
                np.asarray(p.point2D_idxs)[keep],
            )
    if use_init_points:
        pc = read_ply(dataset_dir / manifest.initialization.file)
        if pc.frame not in ("LOCAL_METRIC", "UNKNOWN"):
            raise ContractError(f"init_points.ply must be LOCAL_METRIC, got {pc.frame}")
        if chunk is not None:
            # Chunking cuts the init by where the points are, so where they are must be known:
            # an UNKNOWN-frame PLY projected onto the LOCAL_METRIC axis would select by accident.
            if pc.frame != "LOCAL_METRIC":
                raise ContractError(
                    f"{manifest.initialization.file} declares frame {pc.frame}; a chunk's init is "
                    "selected by chainage on the LOCAL_METRIC axis, so the init must say it is "
                    "LOCAL_METRIC"
                )
            pc = _chunk_init(dataset_dir, manifest, pc, chunk, chunk_evidence)
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
    # Upstream rescales every camera's K by the first image's actual/expected size ratio
    # (colmap.py:262-273), so an image that is not its camera's size would silently train on
    # distorted intrinsics. The Phase 1 render gate refuses the same thing after training.
    from minegs.eval.surface.render import require_images_match_cameras

    require_images_match_cameras(dataset_dir, model.cameras, images)
    if data_factor > 1:
        # Upstream divides K and the image size by data_factor, then rescales every camera by
        # the first image's actual/expected ratio (colmap.py:107, 262-273). Only when each
        # size divides exactly is that ratio 1 for every camera, so that the trainer's K is the
        # image's K and its image size is the size it renders.
        odd = sorted(
            f"camera {cid} {c.width}x{c.height}"
            for cid, c in cams.items()
            if c.width % data_factor or c.height % data_factor
        )
        if odd:
            raise ContractError(
                f"data_factor {data_factor} does not divide {', '.join(odd)}. Upstream would "
                "train those cameras with a K scaled by another camera's rounding; use a factor "
                "that divides every camera size, or data_factor 1."
            )
    clamped = None
    if clamp_init_rgb:
        lo, hi = APPEARANCE_RGB_RANGE
        clamped = 0
        for p in points.values():
            rgb = np.asarray(p.rgb, dtype=np.int64)
            clamped += int(((rgb < lo) | (rgb > hi)).sum())
            p.rgb = np.clip(rgb, lo, hi).astype(np.uint8)
    staged_model = colmap_io.ColmapModel(cams, images, points)
    colmap_io.write_model(staged_model, staged_dir / "sparse" / "0")
    colmap_io.write_model_binary(staged_model, staged_dir / "sparse" / "0")
    downscale = stage_downscaled(staged_dir, chosen, int(data_factor))
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
        downscale=downscale,
        init_rgb_clamped=clamped,
        chunk=chunk_evidence,
    )


def _chunk_init(dataset_dir: Path, manifest: Manifest, pc, chunk, evidence: dict):
    """The init points the existing support locator places inside the chunk's support.

    Chainage comes from ``Support.locate`` on the dataset centerline in LOCAL_METRIC, the rule
    depth supervision uses. A point it cannot place (off the axis by more than the drift could
    be, or past an end) is in no chunk; how many is recorded. The holdout is not re-read: the
    init file is already free of it (Phase 0C/3).
    """
    from minegs.train.supervision.support import Support, dataset_centerline

    cl = dataset_centerline(dataset_dir, manifest)
    if cl is None:
        raise ContractError("a chunk's init is selected by chainage; the dataset has no axis")
    s, ok = Support(cl).locate(pc.xyz)
    lo, hi = chunk.support_range_m
    keep = ok & (s >= lo) & (s <= hi)
    evidence.update(
        init_points_total=len(pc),
        init_points_located=int(ok.sum()),
        init_points_unlocated=int((~ok).sum()),
        init_points_selected=int(keep.sum()),
    )
    if not keep.any():
        raise ContractError(
            f"chunk {chunk.chunk_id}: no init point lies in its support {lo:g}-{hi:g} m"
        )
    return pc.select(np.flatnonzero(keep))
