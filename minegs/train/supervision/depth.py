"""``DepthSupervisionRecord`` — metric depth evidence for training, kept apart from init (Phase 4).

Upstream gsplat derives depth targets from the COLMAP tracks of ``points3D``. That is the same
array its ``init_type=sfm`` initialises from (``simple_trainer.py:234-236`` and
``colmap.py:411-420`` at v1.5.3). Depth evidence and initial geometry are one object there.
MineGS keeps them apart. Depth supervision is an artifact of its own:

```
<dataset>/supervision/depth/<supervision_id>/
    depth_supervision.json     this record
    samples.npy                image, u, v, depth_m, confidence (structured, no pickle)
```

Its identity is the hash of that directory. Its bytes do not depend on the initialisation, and
its dataset binding does not include it. Changing ``init_points.ply`` therefore neither
rewrites this artifact nor invalidates it, and the reverse holds too (AD-1).

``verify_depth_supervision`` re-derives every sample rather than reading the record. It
back-projects ``(u, v, depth)`` through the dataset camera into ``LOCAL_METRIC``, then asks
the leakage questions of the resulting point and ray (``support.Support``). The record's own
statements are compared with what comes out, never used in its place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel, canonical_json
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import sha256_file, sha256_tree
from minegs.ingest.common import colmap_io
from minegs.train.supervision.support import (
    DEFAULT_MAX_RADIAL_M,
    DEFAULT_RAY_MARGIN_M,
    Support,
    support_ranges,
)

SUPERVISION_DIR = "supervision/depth"
RECORD_FILE = "depth_supervision.json"
SAMPLES_FILE = "samples.npy"

#: One row per depth sample. ``image`` indexes ``DepthSupervisionRecord.images``.
SAMPLE_DTYPE = np.dtype(
    [
        ("image", "<u4"),
        ("u", "<f4"),
        ("v", "<f4"),
        ("depth_m", "<f4"),
        ("confidence", "<f4"),
    ]
)

# The contract values (docs/PHASE4_CONTRACT.md AD-2). Kept as plain strings in the record and
# compared here, so a record that says something else is refused with a reason rather than
# failing to parse.
FRAME = "LOCAL_METRIC"
DEPTH_UNIT = "m"
DEPTH_SEMANTICS = "camera_z"
PIXEL_CONVENTION = "colmap_continuous"
BINARY = "binary_mask"
WEIGHT = "unit_interval_weight"
CONFIDENCE_SEMANTICS = (BINARY, WEIGHT)
#: Sources a Phase 4 builder produces. ``sensor_depth`` is a reserved name with no builder and
#: no defined semantics; it is refused rather than half-supported (contract §5.3).
SUPPORTED_SOURCES = ("sfm_tracks", "tls_projection")
SourceKind = Literal["sfm_tracks", "tls_projection", "sensor_depth"]
IMAGE_ONLY_SOURCES = ("video", "video360")
PINHOLE_MODELS = ("PINHOLE", "SIMPLE_PINHOLE")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetBinding(_Strict):
    """What the samples depend on: cameras, poses, split, axis and local origin. Not init."""

    cameras_sha256: str
    poses_sha256: str
    split_sha256: str
    centerline_sha256: str | None
    T_tls_from_local: list[list[float]]
    binding_sha256: str


class SourceAssetRef(_Strict):
    role: str
    path: str
    sha256: str


class DepthSupervisionRecord(VersionedModel):
    """``depth_supervision.json`` (docs/PHASE4_CONTRACT.md §4)."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    supervision_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_binding: DatasetBinding
    source_kind: SourceKind
    source_assets: list[SourceAssetRef] = Field(default_factory=list)
    frame: str
    depth_unit: str
    depth_semantics: str
    pixel_convention: str
    confidence_semantics: str
    images: list[str]
    n_samples: int = Field(ge=0)
    n_samples_in_loss: int = Field(ge=0)
    per_image_counts: list[int]
    support_ranges_m: list[tuple[float, float]] | None = None
    excluded_holdout_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    exclusions: dict[str, int] = Field(default_factory=dict)
    creation_params: dict[str, Any] = Field(default_factory=dict)
    minegs_version: str
    git_commit: str
    created_at: str
    samples_file: str = SAMPLES_FILE
    samples_sha256: str


# ---------------------------------------------------------------- dataset binding


def _digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def dataset_binding(
    dataset_dir: str | Path, manifest: Manifest, model: colmap_io.ColmapModel
) -> DatasetBinding:
    """Digest the parts of a dataset that the samples depend on, and nothing else.

    Poses are digested from the parsed model, not from ``images.txt``: an image-only dataset's
    ``images.txt`` also carries 2D tracks into ``points3D``, which is initialisation, and a
    binding that moved with the init would make depth evidence stale whenever the init changed.
    """
    used = {im.camera_id for im in model.images.values()}
    cameras = {
        str(c.id): [c.model, int(c.width), int(c.height), [float(p) for p in c.params]]
        for c in sorted(model.cameras.values(), key=lambda c: c.id)
        if c.id in used
    }
    poses = {
        im.name: [int(im.camera_id), [float(q) for q in im.qvec], [float(t) for t in im.tvec]]
        for im in sorted(model.images.values(), key=lambda i: i.name)
    }
    split = {
        "split": manifest.split.model_dump(mode="json"),
        "groups": {
            gid: {"members": list(g.members), "span": list(g.span()) if g.span() else None}
            for gid, g in sorted(manifest.capture_groups.items())
        },
    }
    cl = None
    if manifest.centerline is not None:
        p = Path(dataset_dir) / manifest.centerline.file
        cl = sha256_file(p) if p.is_file() else None
    T = [list(map(float, row)) for row in manifest.coordinate_frames.T_tls_from_local]
    parts = {
        "cameras_sha256": _digest(cameras),
        "poses_sha256": _digest(poses),
        "split_sha256": _digest(split),
        "centerline_sha256": cl,
        "T_tls_from_local": T,
    }
    return DatasetBinding(**parts, binding_sha256=_digest(parts))


# ---------------------------------------------------------------- geometry shared by both ends


def backproject_samples(
    samples: np.ndarray, images: list[str], model: colmap_io.ColmapModel
) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v, depth)`` -> LOCAL_METRIC point and its camera centre, one row per sample.

    ``u, v`` are COLMAP continuous pixel coordinates (pixel centre at +0.5) and ``depth`` is
    camera +z, so ``X_cam = ((u - cx) / fx * z, (v - cy) / fy * z, z)``. The builder calls this
    on its own float32 output, so the builder and the verifier derive support from the same
    numbers.
    """
    by_name = model.image_by_name()
    pts = np.zeros((len(samples), 3))
    cams = np.zeros((len(samples), 3))
    idx = samples["image"].astype(np.int64)
    for k in np.unique(idx):
        im = by_name[images[int(k)]]
        K = model.cameras[im.camera_id].K()
        sel = idx == k
        z = samples["depth_m"][sel].astype(np.float64)
        x = (samples["u"][sel].astype(np.float64) - K[0, 2]) / K[0, 0] * z
        y = (samples["v"][sel].astype(np.float64) - K[1, 2]) / K[1, 1] * z
        pts[sel] = im.world_from_cam.apply(np.column_stack([x, y, z]))
        cams[sel] = im.center
    return pts, cams


def recorded_support(
    samples: np.ndarray, images: list[str], model: colmap_io.ColmapModel, support: Support
) -> list[tuple[float, float]] | None:
    """The support ranges a record states, re-derived from its samples."""
    if support.centerline is None:
        return None
    pts, _ = backproject_samples(samples, images, model)
    s, ok = support.locate(pts)
    return support_ranges(s[ok])


# ---------------------------------------------------------------- verification


@dataclass
class VerifiedDepthSupervision:
    """An artifact that passed every check, with the numbers a run records about it."""

    path: Path
    record: DepthSupervisionRecord
    samples: np.ndarray
    artifact_sha256: str
    record_sha256: str

    def summary(self) -> dict[str, Any]:
        r = self.record
        return {
            "supervision_id": r.supervision_id,
            "source_kind": r.source_kind,
            "artifact_sha256": self.artifact_sha256,
            "record_sha256": self.record_sha256,
            "samples_sha256": r.samples_sha256,
            "n_samples": r.n_samples,
            "n_samples_in_loss": r.n_samples_in_loss,
            "n_images": len(r.images),
            "confidence_semantics": r.confidence_semantics,
            "depth_semantics": r.depth_semantics,
            "depth_unit": r.depth_unit,
            "frame": r.frame,
            "pixel_convention": r.pixel_convention,
            "dataset_binding_sha256": r.dataset_binding.binding_sha256,
        }

    def images_with_samples(self, names: list[str] | None = None) -> list[str]:
        """Images carrying at least one sample that enters the loss."""
        counts = np.bincount(
            self.samples["image"][self.samples["confidence"] > 0].astype(np.int64),
            minlength=len(self.record.images),
        )
        have = [n for n, c in zip(self.record.images, counts, strict=True) if c > 0]
        if names is None:
            return have
        wanted = set(names)
        return [n for n in have if n in wanted]


def _refuse(artifact: Path, why: str) -> ContractError:
    return ContractError(f"depth supervision {artifact}: {why}")


def _load_samples(path: Path, artifact: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise _refuse(artifact, f"{path.name} is not a readable sample array ({e})") from e
    if arr.dtype != SAMPLE_DTYPE or arr.ndim != 1:
        raise _refuse(
            artifact,
            f"{path.name} has dtype {arr.dtype} and shape {arr.shape}; the contract is a 1-D "
            f"array of {SAMPLE_DTYPE}",
        )
    return arr


def verify_depth_supervision(
    dataset_dir: str | Path,
    artifact_dir: str | Path,
    *,
    manifest: Manifest | None = None,
    model: colmap_io.ColmapModel | None = None,
) -> VerifiedDepthSupervision:
    """Re-derive an artifact against a dataset; refuse on the first thing that does not hold.

    Nothing here is a warning. Each check corresponds to a row of docs/PHASE4_CONTRACT.md §7.
    """
    ds = Path(dataset_dir)
    art = Path(artifact_dir)
    rec_path = art / RECORD_FILE
    if not rec_path.is_file():
        raise _refuse(art, f"no {RECORD_FILE}; this is not a depth supervision artifact")
    record = DepthSupervisionRecord.load(rec_path)
    manifest = manifest or Manifest.load_dataset(ds)
    model = model or colmap_io.read_model(ds / "sparse" / "0")

    # ---- what the samples mean
    if record.source_kind not in SUPPORTED_SOURCES:
        raise _refuse(
            art,
            f"source_kind {record.source_kind!r} has no Phase 4 builder; its depth semantics "
            f"and uncertainty are undefined, so it is not accepted (supported: "
            f"{SUPPORTED_SOURCES})",
        )
    for field_name, have, want in (
        ("frame", record.frame, FRAME),
        ("depth_unit", record.depth_unit, DEPTH_UNIT),
        ("depth_semantics", record.depth_semantics, DEPTH_SEMANTICS),
        ("pixel_convention", record.pixel_convention, PIXEL_CONVENTION),
    ):
        if have != want:
            raise _refuse(
                art,
                f"{field_name} is {have!r}; training consumes {want!r} only. Depth in another "
                "frame, unit or meaning would be compared with the rendered camera-z in metres "
                "as if it were the same quantity.",
            )
    if record.confidence_semantics not in CONFIDENCE_SEMANTICS:
        raise _refuse(
            art,
            f"confidence_semantics {record.confidence_semantics!r} is not one of "
            f"{CONFIDENCE_SEMANTICS}",
        )
    if record.source_kind == "tls_projection" and manifest.source in IMAGE_ONLY_SOURCES:
        raise _refuse(
            art,
            f"TLS-projected depth on a {manifest.source} dataset. The image-only path is "
            "defined by not training on scanner geometry; supervising it with TLS depth would "
            "make it TLS-assisted while its manifest says otherwise (Phase 3 independence).",
        )

    # ---- the bytes are the bytes the record names
    if Path(record.samples_file).name != record.samples_file or record.samples_file in (
        "",
        ".",
        "..",
    ):
        raise _refuse(art, f"samples_file {record.samples_file!r} must be a file in the artifact")
    samples_path = art / record.samples_file
    if not samples_path.is_file():
        raise _refuse(art, f"{record.samples_file} is missing")
    # The artifact's identity is the hash of its directory, and the trainer receives the whole
    # directory. A file in it that the record does not name would travel with that identity
    # without ever being checked, so it is refused (as Phase 3 refuses frame-set stowaways).
    named = {art / RECORD_FILE, art / record.samples_file}
    extra = sorted(
        str(p.relative_to(art)) for p in art.rglob("*") if p.is_file() and p not in named
    )
    if extra:
        raise _refuse(art, f"holds files its record does not name: {extra[:5]}")
    got = sha256_file(samples_path)
    if got != record.samples_sha256:
        raise _refuse(
            art,
            f"{record.samples_file} hashes to {got[:12]}, but the record names "
            f"{record.samples_sha256[:12]}: the samples were changed after the record was "
            "written",
        )
    samples = _load_samples(samples_path, art)

    # ---- the dataset is the dataset the samples were made against
    if record.dataset_id != manifest.dataset_id:
        raise _refuse(
            art,
            f"made for dataset {record.dataset_id!r}, not {manifest.dataset_id!r}",
        )
    binding = dataset_binding(ds, manifest, model)
    if binding.binding_sha256 != record.dataset_binding.binding_sha256:
        differ = [
            k
            for k in ("cameras_sha256", "poses_sha256", "split_sha256", "centerline_sha256")
            if getattr(binding, k) != getattr(record.dataset_binding, k)
        ]
        if binding.T_tls_from_local != record.dataset_binding.T_tls_from_local:
            differ.append("T_tls_from_local")
        raise _refuse(
            art,
            f"the dataset no longer matches the one the samples were made against "
            f"(differs: {', '.join(differ) or 'binding digest'}). Depth measured through other "
            "cameras, or against another split, is not this dataset's evidence.",
        )

    # ---- images: training images of this dataset, and nothing else
    names = list(record.images)
    if len(set(names)) != len(names):
        raise _refuse(art, "images lists a name twice")
    by_name = model.image_by_name()
    absent = [n for n in names if n not in by_name]
    if absent:
        raise _refuse(art, f"{len(absent)} image(s) are not in this dataset (e.g. {absent[:3]})")
    train = set(manifest.train_images())
    test = set(manifest.test_images())
    held = [n for n in names if n not in train or n in test]
    if held:
        raise _refuse(
            art,
            f"{len(held)} image(s) are not training images of this dataset (e.g. {held[:3]}). "
            "A held-out image cannot supply supervision.",
        )
    for n in names:
        cam = model.cameras[by_name[n].camera_id]
        if cam.model not in PINHOLE_MODELS:
            raise _refuse(
                art,
                f"{n} uses camera model {cam.model}; depth supervision, like the renderer, is "
                f"defined for {PINHOLE_MODELS} only",
            )

    # ---- every sample is a finite, positive, in-frame, correctly weighted observation
    if len(samples) != record.n_samples:
        raise _refuse(art, f"{len(samples)} samples on disk, record says {record.n_samples}")
    idx = samples["image"].astype(np.int64)
    if len(samples) and int(idx.max()) >= len(names):
        raise _refuse(art, "a sample names an image index outside the record's image list")
    for f in ("u", "v", "depth_m", "confidence"):
        if not np.all(np.isfinite(samples[f])):
            raise _refuse(art, f"non-finite {f} in the samples")
    if np.any(samples["depth_m"] <= 0):
        raise _refuse(art, "a sample has depth <= 0; camera-z of a visible surface is positive")
    conf = samples["confidence"]
    if record.confidence_semantics == BINARY:
        bad = ~np.isin(conf, (0.0, 1.0))
        if np.any(bad):
            raise _refuse(
                art,
                f"confidence_semantics is {BINARY} but {int(bad.sum())} value(s) are neither 0 "
                f"nor 1 (e.g. {float(conf[bad][0])})",
            )
    elif np.any((conf < 0) | (conf > 1)):
        raise _refuse(art, f"confidence outside [0, 1] under {WEIGHT}")
    counts = np.bincount(idx, minlength=len(names)).tolist() if len(samples) else [0] * len(names)
    if counts != list(record.per_image_counts):
        raise _refuse(art, "per_image_counts does not match the samples")
    if int((conf > 0).sum()) != record.n_samples_in_loss:
        raise _refuse(art, "n_samples_in_loss does not match the samples")
    for k, n in enumerate(names):
        sel = idx == k
        if not sel.any():
            continue
        cam = model.cameras[by_name[n].camera_id]
        u, v = samples["u"][sel], samples["v"][sel]
        if np.any((u < 0) | (u >= cam.width) | (v < 0) | (v >= cam.height)):
            raise _refuse(art, f"a sample of {n} lies outside its {cam.width}x{cam.height} image")

    # ---- leakage, re-derived from the samples themselves (AD-3)
    holdout = [(float(lo), float(hi)) for lo, hi in record.excluded_holdout_ranges_m]
    # The record's own parameters are used only when they are at least as strict as the
    # contract's: a record that widened what counts as located, or narrowed the ray margin,
    # would otherwise be judged by the leniency it granted itself.
    max_radial = float(record.creation_params.get("max_radial_m", DEFAULT_MAX_RADIAL_M))
    margin = float(record.creation_params.get("ray_margin_m", DEFAULT_RAY_MARGIN_M))
    if not (max_radial <= DEFAULT_MAX_RADIAL_M and margin >= DEFAULT_RAY_MARGIN_M):
        raise _refuse(
            art,
            f"built with max_radial_m={max_radial}, ray_margin_m={margin}; the contract allows "
            f"at most {DEFAULT_MAX_RADIAL_M} and at least {DEFAULT_RAY_MARGIN_M}",
        )
    support = Support.of_dataset(ds, manifest, max_radial_m=max_radial, ray_margin_m=margin)
    if sorted(holdout) != sorted(support.holdout):
        raise _refuse(
            art,
            f"records holdout {holdout} but the dataset declares {support.holdout}; the "
            "exclusion was made against a different holdout",
        )
    if support.required and len(samples):
        pts, cams = backproject_samples(samples, names, model)
        keep, _, reasons = support.classify(cams, pts)
        if not np.all(keep):
            why = {k: int(m.sum()) for k, m in reasons.items() if m.any()}
            raise _refuse(
                art,
                f"{int((~keep).sum())} sample(s) may not train: {why}. A sample in or through "
                f"the holdout {support.holdout}, or one whose place along the drift cannot be "
                "established, is evidence about the geometry the evaluation is measured on.",
            )
    derived = recorded_support(samples, names, model, support)
    want = None if record.support_ranges_m is None else [tuple(r) for r in record.support_ranges_m]
    if not _same_ranges(derived, want):
        raise _refuse(art, f"support_ranges_m {want} does not follow from the samples ({derived})")

    return VerifiedDepthSupervision(
        path=art,
        record=record,
        samples=samples,
        artifact_sha256=sha256_tree(art),
        record_sha256=sha256_file(rec_path),
    )


def _same_ranges(a: list | None, b: list | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    return bool(np.allclose(np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)))


def load_record_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


__all__ = [
    "BINARY",
    "CONFIDENCE_SEMANTICS",
    "DEPTH_SEMANTICS",
    "DEPTH_UNIT",
    "FRAME",
    "PIXEL_CONVENTION",
    "RECORD_FILE",
    "SAMPLES_FILE",
    "SAMPLE_DTYPE",
    "SUPERVISION_DIR",
    "SUPPORTED_SOURCES",
    "WEIGHT",
    "DatasetBinding",
    "DepthSupervisionRecord",
    "SourceAssetRef",
    "VerifiedDepthSupervision",
    "backproject_samples",
    "dataset_binding",
    "recorded_support",
    "verify_depth_supervision",
]
