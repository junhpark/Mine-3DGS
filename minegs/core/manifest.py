"""Dataset contract (§4): ``dataset/manifest.json`` v1.

Required: schema_version, dataset_id, coordinate_frames, capture_groups, split,
initialization, provenance. Optional: source, capture_epoch, scale, registration,
pano_convention, centerline, chunks.

Structural validation lives here (references resolve, transforms are rigid, groups are
disjoint). *Claims* (what an evaluation may say about a run) are judged in
``minegs.eval.protocol`` from the same fields (§5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from minegs.core.chunking import ChunkPlan
from minegs.core.config import MigrationRegistry, VersionedModel
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Sim3
from minegs.core.provenance import SourceAsset

MANIFEST_FILE = "manifest.json"
DATASET_LAYOUT = ("images", "sparse/0", "init_points.ply", "manifest.json")
SPARSE_FILES = ("cameras.txt", "images.txt", "points3D.txt")

GroupType = Literal["tls_station", "trajectory_segment", "camera_rig", "mobile_mapping_segment"]
InitSource = Literal["tls", "sfm_sparse", "random"]
DatasetSource = Literal["tls", "video", "video360"]
ScaleBasis = Literal["tls_pose", "sim3_to_tls", "known_target"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoordinateFrames(_Strict):
    evaluation: Literal["TLS_GLOBAL"] = "TLS_GLOBAL"
    training: Literal["LOCAL_METRIC"] = "LOCAL_METRIC"
    unit: Literal["m"] = "m"
    T_tls_from_local: list[list[float]]

    @field_validator("T_tls_from_local")
    @classmethod
    def _rigid(cls, v: list[list[float]]) -> list[list[float]]:
        try:
            SE3.from_matrix(v)
        except Exception as e:
            raise ValueError(f"T_tls_from_local must be a rigid 4x4 SE(3) matrix: {e}") from e
        return v

    @property
    def se3(self) -> SE3:
        return SE3.from_matrix(self.T_tls_from_local)


class CaptureGroup(_Strict):
    type: GroupType
    members: list[str] = Field(min_length=1)
    chainage_m: float | None = None
    chainage_range_m: tuple[float, float] | None = None
    pano_id: str | None = None  # E57 station <-> panorama contract (§6.1)

    @model_validator(mode="after")
    def _chainage(self) -> CaptureGroup:
        if (
            self.chainage_range_m is not None
            and self.chainage_range_m[0] > self.chainage_range_m[1]
        ):
            raise ValueError("chainage_range_m must be [lo, hi]")
        return self

    def span(self) -> tuple[float, float] | None:
        if self.chainage_range_m is not None:
            return self.chainage_range_m
        if self.chainage_m is not None:
            return (self.chainage_m, self.chainage_m)
        return None


class GeometryHoldout(_Strict):
    chainage_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    points_excluded: bool = True  # init_points 에서 제외
    images_excluded: bool = False  # false = 복원 시험, true = 외삽 시험

    @field_validator("chainage_ranges_m")
    @classmethod
    def _ordered(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for lo, hi in v:
            if hi <= lo:
                raise ValueError(f"holdout range must be [lo, hi], got {(lo, hi)}")
        return v


class Split(_Strict):
    train_groups: list[str] = Field(default_factory=list)
    test_groups: list[str] = Field(default_factory=list)
    geometry_holdout: GeometryHoldout | None = None

    @model_validator(mode="after")
    def _disjoint(self) -> Split:
        overlap = set(self.train_groups) & set(self.test_groups)
        if overlap:
            raise ValueError(f"groups in both train and test: {sorted(overlap)}")
        return self


class Initialization(_Strict):
    source: InitSource
    file: str = "init_points.ply"
    groups: list[str] = Field(default_factory=list)
    excluded_chainage_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    n_points: int | None = None


class ManifestProvenance(_Strict):
    minegs_version: str
    git_commit: str = "unknown"
    config_hash: str = ""
    source_assets: list[SourceAsset] = Field(default_factory=list)
    created_at: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)


class CaptureEpoch(_Strict):
    id: str
    date: str | None = None


class Scale(_Strict):
    basis: ScaleBasis
    factor: float = 1.0


class Registration(_Strict):
    method: str
    scale: float
    rmse_m: float
    inlier_ratio: float = Field(ge=0.0, le=1.0)
    transform: list[list[float]]  # Sim3 or SE3 4x4, TLS_GLOBAL <- SfM source frame
    n_correspondences: int | None = None
    inlier_threshold_m: float | None = None

    @property
    def sim3(self) -> Sim3:
        return Sim3.from_matrix(self.transform)


class PanoConvention(_Strict):
    az_sign: Literal[1, -1] = 1
    el_flip: bool = False
    az_offset: float = 0.0  # degrees
    source: str = "E57Embedded"
    vendor: str | None = None


class CenterlineRef(_Strict):
    file: str
    source: Literal["design", "extracted"]
    frame: Literal["TLS_GLOBAL", "LOCAL_METRIC"] = "TLS_GLOBAL"


MANIFEST_MIGRATIONS = MigrationRegistry("manifest")


@MANIFEST_MIGRATIONS.register("0.9", "1.0")
def _m_0_9_to_1_0(d: dict[str, Any]) -> dict[str, Any]:
    """Pre-v2 drafts used ``stations`` (TLS only) and a station-level holdout."""
    if "stations" in d and "capture_groups" not in d:
        d["capture_groups"] = {
            sid: {
                "type": "tls_station",
                "members": st.get("images", []),
                "chainage_m": st.get("chainage_m"),
            }
            for sid, st in d.pop("stations").items()
        }
    if "coordinate_frames" not in d:
        d["coordinate_frames"] = {"T_tls_from_local": np.eye(4).tolist()}
    split = d.setdefault("split", {})
    if "holdout_stations" in split:  # station-level holdout is not leak-proof (§4) -> drop
        split.pop("holdout_stations")
    d.setdefault("initialization", {"source": "tls", "groups": split.get("train_groups", [])})
    d.setdefault("provenance", {"minegs_version": "0.0.0"})
    return d


class Manifest(VersionedModel):
    """``dataset/manifest.json``. Load with ``Manifest.load(path)`` (migrates), or
    ``Manifest.load_dataset(dataset_dir)`` which also checks the on-disk layout."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"
    MIGRATIONS: ClassVar[MigrationRegistry | None] = MANIFEST_MIGRATIONS

    dataset_id: str = Field(min_length=1)
    coordinate_frames: CoordinateFrames
    capture_groups: dict[str, CaptureGroup]
    split: Split
    initialization: Initialization
    provenance: ManifestProvenance

    source: DatasetSource | None = None
    capture_epoch: CaptureEpoch | None = None
    scale: Scale | None = None
    registration: Registration | None = None
    pano_convention: PanoConvention | None = None
    centerline: CenterlineRef | None = None
    chunks: ChunkPlan | None = None

    # ---------------------------------------------------------------- validation
    @model_validator(mode="after")
    def _references(self) -> Manifest:
        groups = set(self.capture_groups)
        if not groups:
            raise ValueError("capture_groups must not be empty")
        for field_name, lst in (
            ("split.train_groups", self.split.train_groups),
            ("split.test_groups", self.split.test_groups),
            ("initialization.groups", self.initialization.groups),
        ):
            unknown = sorted(set(lst) - groups)
            if unknown:
                raise ValueError(f"{field_name} references unknown groups {unknown}")
        seen: dict[str, str] = {}
        for gid, g in self.capture_groups.items():
            for m in g.members:
                if m in seen:
                    raise ValueError(f"image {m!r} belongs to groups {seen[m]!r} and {gid!r}")
                seen[m] = gid
        if self.initialization.source == "tls" and not self.initialization.groups:
            raise ValueError("initialization.source=tls requires initialization.groups")
        if self.chunks is not None:
            for c in self.chunks.items:
                unknown = sorted(set(c.groups) - groups)
                if unknown:
                    raise ValueError(f"chunk {c.id} references unknown groups {unknown}")
        return self

    # ---------------------------------------------------------------- accessors
    @property
    def T_tls_from_local(self) -> SE3:
        return self.coordinate_frames.se3

    @property
    def T_local_from_tls(self) -> SE3:
        return self.coordinate_frames.se3.inverse()

    def all_images(self) -> list[str]:
        return [m for g in self.capture_groups.values() for m in g.members]

    def images_of(self, groups: list[str]) -> list[str]:
        return [m for gid in groups for m in self.capture_groups[gid].members]

    def group_of(self, image: str) -> str:
        for gid, g in self.capture_groups.items():
            if image in g.members:
                return gid
        raise KeyError(image)

    def train_images(self) -> list[str]:
        imgs = self.images_of(self.split.train_groups)
        ho = self.split.geometry_holdout
        if ho and ho.images_excluded and ho.chainage_ranges_m:
            keep = []
            for gid in self.split.train_groups:
                span = self.capture_groups[gid].span()
                if span is None or not _intersects_any(span, ho.chainage_ranges_m):
                    keep.extend(self.capture_groups[gid].members)
            imgs = keep
        return imgs

    def test_images(self) -> list[str]:
        return self.images_of(self.split.test_groups)

    def group_chainage(self) -> dict[str, tuple[float, float]]:
        return {
            gid: span for gid, g in self.capture_groups.items() if (span := g.span()) is not None
        }

    def chainage_extent(self) -> tuple[float, float] | None:
        spans = list(self.group_chainage().values())
        if not spans:
            return None
        return (min(s[0] for s in spans), max(s[1] for s in spans))

    # ---------------------------------------------------------------- consistency
    def consistency_issues(self) -> list[str]:
        """Non-fatal contract issues (leaks, missing metadata). Empty list == clean."""
        issues: list[str] = []
        init = self.initialization
        split = self.split
        leak = sorted(set(init.groups) & set(split.test_groups))
        if leak:
            issues.append(f"initialization uses test groups {leak}: novel_view claims are void")
        ho = split.geometry_holdout
        if ho and ho.chainage_ranges_m:
            if not ho.points_excluded:
                issues.append(
                    "geometry_holdout.points_excluded=false: holdout geometry leaks via init"
                )
            missing = [
                r for r in ho.chainage_ranges_m if not _covered(r, init.excluded_chainage_ranges_m)
            ]
            if missing:
                issues.append(
                    f"holdout ranges {missing} are not in initialization.excluded_chainage_ranges_m"
                )
            if init.source == "tls":
                for gid in init.groups:
                    span = self.capture_groups[gid].span()
                    if span is None:
                        issues.append(
                            f"init group {gid} has no chainage: cannot prove holdout exclusion"
                        )
        if self.scale is None:
            issues.append("scale.basis missing: geometry evaluation will refuse to run")
        if self.source in ("video", "video360") and self.registration is None:
            issues.append("video source without registration: metric claims void")
        if self.registration is not None and self.registration.inlier_ratio == 0:
            issues.append("registration.inlier_ratio == 0")
        return issues

    # ---------------------------------------------------------------- dataset io
    @classmethod
    def load_dataset(cls, dataset_dir: str | Path, strict_layout: bool = True) -> Manifest:
        dataset_dir = Path(dataset_dir)
        mpath = dataset_dir / MANIFEST_FILE
        if not mpath.exists():
            raise ContractError(f"{dataset_dir}: no {MANIFEST_FILE}")
        m = cls.load(mpath)
        if strict_layout:
            problems = validate_layout(dataset_dir, m)
            if problems:
                raise ContractError(f"{dataset_dir}: " + "; ".join(problems))
        return m

    def save_dataset(self, dataset_dir: str | Path) -> Path:
        return self.save(Path(dataset_dir) / MANIFEST_FILE)


def _intersects_any(span: tuple[float, float], ranges: list[tuple[float, float]]) -> bool:
    return any(span[1] >= lo and span[0] <= hi for lo, hi in ranges)


def _covered(r: tuple[float, float], ranges: list[tuple[float, float]]) -> bool:
    return any(lo <= r[0] and hi >= r[1] for lo, hi in ranges)


def validate_layout(dataset_dir: str | Path, manifest: Manifest | None = None) -> list[str]:
    """Check the on-disk dataset contract (§4). Returns a list of problems."""
    d = Path(dataset_dir)
    problems: list[str] = []
    if not (d / "images").is_dir():
        problems.append("missing images/")
    for f in SPARSE_FILES:
        if not (d / "sparse" / "0" / f).exists():
            problems.append(f"missing sparse/0/{f}")
    if manifest is not None:
        init = d / manifest.initialization.file
        if not init.exists():
            problems.append(f"missing {manifest.initialization.file}")
        imgs = d / "images"
        if imgs.is_dir():
            missing = [m for m in manifest.all_images() if not (imgs / m).exists()]
            if missing:
                problems.append(
                    f"{len(missing)} manifest images missing on disk (e.g. {missing[:3]})"
                )
        if manifest.centerline is not None:
            cl = d / manifest.centerline.file
            if not cl.exists() and not (d.parent / manifest.centerline.file).exists():
                problems.append(f"centerline file {manifest.centerline.file} not found")
    return problems
