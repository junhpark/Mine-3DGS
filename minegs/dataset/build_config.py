"""Dataset build configuration (Phase 0C §22): versioned, strict, extra-forbid.

Every result-changing decision of a build lives here rather than in scattered CLI flags, so
``provenance.config_hash`` names *the* configuration and the same staging tree with the same
config yields the same dataset. The resolved form (with the origin actually chosen, the
convention actually used) is written into the dataset as ``build_config.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.ingest.common.equirect import RingCropSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceFrameMode = Literal["explicit_identity", "explicit_transform"]


class SourceFrameConfig(_Strict):
    """How the E57's own SOURCE frame becomes TLS_GLOBAL (§5).

    There is no default and no inference. ``explicit_identity`` is the user *declaring* that
    this file's registered frame is the survey reference; ``explicit_transform`` supplies the
    SE(3). Either way the declaration is recorded, and an identity that nobody declared never
    happens by omission.
    """

    mode: SourceFrameMode
    T_tls_from_source: list[list[float]] | None = None
    #: Where the transform came from (a registration report, a survey control sheet, ...).
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SourceFrameConfig:
        if self.mode == "explicit_transform":
            if self.T_tls_from_source is None:
                raise ValueError("source_frame.mode=explicit_transform needs T_tls_from_source")
            try:
                SE3.from_matrix(self.T_tls_from_source)
            except Exception as e:
                raise ValueError(f"T_tls_from_source must be a rigid SE(3) 4x4: {e}") from e
        elif self.T_tls_from_source is not None:
            raise ValueError(
                "source_frame.mode=explicit_identity must not also carry T_tls_from_source; "
                "use mode=explicit_transform for a non-identity transform"
            )
        return self

    def se3(self) -> SE3:
        if self.mode == "explicit_identity":
            return SE3.identity()
        return SE3.from_matrix(self.T_tls_from_source)


OriginPolicy = Literal["station_centroid_rounded", "explicit"]


class LocalMetricConfig(_Strict):
    """LOCAL_METRIC origin (§7): translation only, scale 1, deterministic."""

    origin_policy: OriginPolicy = "station_centroid_rounded"
    rounding_m: float = Field(default=0.1, gt=0)
    origin_tls: list[float] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> LocalMetricConfig:
        if self.origin_policy == "explicit":
            if self.origin_tls is None or len(self.origin_tls) != 3:
                raise ValueError("local_metric.origin_policy=explicit needs origin_tls [x, y, z]")
            if not np.all(np.isfinite(self.origin_tls)):
                raise ValueError("local_metric.origin_tls must be finite")
        elif self.origin_tls is not None:
            raise ValueError("origin_tls is only used with origin_policy=explicit")
        return self


class PanoConventionConfig(_Strict):
    az_sign: Literal[1, -1] = 1
    el_flip: bool = False
    az_offset_deg: float = 0.0
    source: str = "E57Embedded"
    vendor: str | None = None


CameraMode = Literal["e57_pinhole", "e57_spherical"]


class CameraConfig(_Strict):
    """Camera path (§8–§13).

    ``e57_pinhole`` needs the E57-camera → COLMAP-camera axis convention, from a calibration
    artifact (``convention_file``) or written out explicitly (``R_e57cam_from_cam``). Neither
    is defaulted: a convention nobody declared or measured is a guess, and a wrong one trains
    for hours before anyone notices.
    """

    mode: CameraMode
    convention_file: str | None = None
    R_e57cam_from_cam: list[list[float]] | None = None
    pano_convention: PanoConventionConfig | None = None
    ring_crop: RingCropSpec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CameraConfig:
        if self.mode == "e57_pinhole":
            if (self.convention_file is None) == (self.R_e57cam_from_cam is None):
                raise ValueError(
                    "camera.mode=e57_pinhole needs exactly one of convention_file (a "
                    "calibrate-camera artifact) or R_e57cam_from_cam (an explicit 3x3)"
                )
            if self.pano_convention is not None or self.ring_crop is not None:
                raise ValueError("pano_convention/ring_crop belong to camera.mode=e57_spherical")
        else:
            if self.ring_crop is None:
                raise ValueError("camera.mode=e57_spherical needs ring_crop")
            if self.convention_file is not None or self.R_e57cam_from_cam is not None:
                raise ValueError(
                    "convention_file/R_e57cam_from_cam belong to camera.mode=e57_pinhole"
                )
        return self


class SplitConfig(_Strict):
    """Group-level split (§19). Nothing here means reconstruction-only."""

    test_every: int | None = Field(default=None, ge=2)
    train_groups: list[str] | None = None
    test_groups: list[str] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SplitConfig:
        explicit = self.train_groups is not None or self.test_groups is not None
        if self.test_every is not None and explicit:
            raise ValueError("split: use either test_every or explicit train/test groups")
        if (self.train_groups is None) != (self.test_groups is None):
            raise ValueError("split: explicit train_groups and test_groups go together")
        if explicit:
            both = set(self.train_groups or []) & set(self.test_groups or [])
            if both:
                raise ValueError(f"split: groups in both train and test: {sorted(both)}")
        return self

    @property
    def requested(self) -> bool:
        return self.test_every is not None or self.test_groups is not None


class GeometryHoldoutConfig(_Strict):
    ranges_m: list[tuple[float, float]] = Field(min_length=1)
    images_excluded: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> GeometryHoldoutConfig:
        for lo, hi in self.ranges_m:
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                raise ValueError(f"holdout range must be finite [lo, hi], got {(lo, hi)}")
        return self


class CenterlineConfig(_Strict):
    """A design/measured centerline file (§18 A) or an extraction from the TLS cloud (§18 B).

    Extraction reuses ``Centerline.extract_from_points`` and inherits its limitation: a
    straight-ish drift. It is recorded as ``source=extracted`` and never promoted to design.
    """

    file: str | None = None
    frame: Literal["SOURCE", "TLS_GLOBAL"] = "TLS_GLOBAL"
    extract_from_tls: bool = False
    bin_m: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> CenterlineConfig:
        if (self.file is None) == (not self.extract_from_tls):
            raise ValueError("centerline: give exactly one of file or extract_from_tls=true")
        return self


class InitializationConfig(_Strict):
    voxel_m: float | None = Field(default=0.02, gt=0)
    max_points: int = Field(default=1_000_000, ge=1)
    #: ``sparse/0/points3D.txt`` gets a subsample of the *same* leak-free set (§20).
    sparse_max_points: int = Field(default=200_000, ge=0)
    seed: int = 0


class CaptureEpochConfig(_Strict):
    id: str
    date: str | None = None


class ImagesConfig(_Strict):
    #: ``hardlink`` falls back to a copy across filesystems; ``copy`` always copies.
    mode: Literal["hardlink", "copy"] = "hardlink"


class DatasetBuildConfig(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    dataset_id: str = Field(min_length=1)
    source_frame: SourceFrameConfig
    local_metric: LocalMetricConfig = Field(default_factory=LocalMetricConfig)
    camera: CameraConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    geometry_holdout: GeometryHoldoutConfig | None = None
    centerline: CenterlineConfig | None = None
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    capture_epoch: CaptureEpochConfig | None = None
    images: ImagesConfig = Field(default_factory=ImagesConfig)

    @model_validator(mode="after")
    def _holdout_needs_centerline(self) -> DatasetBuildConfig:
        if self.geometry_holdout is not None and self.centerline is None:
            raise ValueError(
                "geometry_holdout needs a centerline: holdout ranges are chainage, and without "
                "a line there is no chainage (scan order is not chainage, §17)"
            )
        return self

    def resolve_paths(self, base: Path) -> DatasetBuildConfig:
        """Make relative file references absolute against the config file's directory."""

        def fix(p: str | None) -> str | None:
            if p is None:
                return None
            q = Path(p)
            return str(q if q.is_absolute() else (base / q).resolve())

        cfg = self.model_copy(deep=True)
        cfg.camera.convention_file = fix(cfg.camera.convention_file)
        if cfg.centerline is not None:
            cfg.centerline.file = fix(cfg.centerline.file)
        return cfg


def load_build_config(path: str | Path) -> DatasetBuildConfig:
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"build config {p} not found")
    return DatasetBuildConfig.load(p).resolve_paths(p.parent)


def example_config() -> dict[str, Any]:
    """A complete example (README §Phase 0C). ``T_tls_from_source`` is the user's declaration."""
    return {
        "schema_version": "1.0",
        "dataset_id": "mine_tunnel_ep1",
        "source_frame": {
            "mode": "explicit_identity",
            "note": "registered Matterport survey frame adopted as the TLS_GLOBAL reference",
        },
        "local_metric": {"origin_policy": "station_centroid_rounded", "rounding_m": 0.1},
        "camera": {"mode": "e57_pinhole", "convention_file": "camera_convention.json"},
        "split": {"test_every": 5},
        "centerline": {"file": "centerline.csv", "frame": "TLS_GLOBAL"},
        "geometry_holdout": {"ranges_m": [[40.0, 50.0]], "images_excluded": False},
        "initialization": {"voxel_m": 0.02, "max_points": 1_000_000},
        "capture_epoch": {"id": "ep1"},
    }
