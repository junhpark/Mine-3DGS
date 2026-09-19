"""The metric surface artifact (§1.7, Phase 1A).

A run's ``point_cloud/*.ply`` holds Gaussian *centres*: the parameters of a volumetric
radiance field, not samples of the tunnel wall. Comparing them against TLS measures how the
optimiser distributed its primitives, not how accurate the reconstruction is. So a surface has
to be *derived* — depth maps today, TSDF/mesh later — and the derivation has to leave a record.

``SurfaceRecord`` is that record, and nothing more. It says which run and which dataset the
samples came from, by what method, in what frame, and how many there are. It states no
accuracy: a surface artifact is what makes a geometry claim *possible*, never what makes one
true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import ProvenanceRecord

SURFACE_FILE = "surface.json"
SURFACE_POINTS_FILE = "surface_points.ply"
SURFACE_DIRNAME = "surface"

#: Phase 1A ships one method. TSDF and mesh extraction are later phases (docs/ROADMAP.md);
#: adding them here before they exist would let a record name a provenance nothing can produce.
SurfaceMethod = Literal["depth_backprojection"]


class SurfaceRecord(VersionedModel):
    """``<surface_dir>/surface.json``."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    surface_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    method: SurfaceMethod
    # Not configurable. Surface samples are evaluated after a rigid hop into TLS_GLOBAL (§3),
    # which is only defined if they arrive in the dataset's metric training frame.
    frame: Literal["LOCAL_METRIC"] = "LOCAL_METRIC"
    unit: Literal["m"] = "m"
    #: Relative to the directory holding this file, so the artifact survives being moved.
    point_file: str = SURFACE_POINTS_FILE
    point_count: int = Field(gt=0)
    depth_map_count: int = Field(gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Cheap sanity, not geometry: an artifact whose span is centimetres or kilometres is worth
    # noticing before it reaches an evaluator. Nothing reads these to decide anything.
    bounds_min_m: list[float] | None = None
    bounds_max_m: list[float] | None = None
    span_m: list[float] | None = None
    provenance: ProvenanceRecord

    @field_validator("point_file")
    @classmethod
    def _relative_name(cls, v: str) -> str:
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"point_file must be relative to the surface directory, got {v!r}")
        return v

    def points_path(self, surface_dir: str | Path) -> Path:
        return Path(surface_dir) / self.point_file


def find_surface(path: str | Path) -> Path | None:
    """``surface.json`` for *path*, or ``None`` when *path* is not a surface artifact.

    Accepts the directory (the documented form) or the file itself. Returning ``None`` rather
    than raising keeps the caller free to decide what a non-artifact means: a refusal on the
    claim-bearing path, a raw PLY on the diagnostic one.
    """
    p = Path(path)
    if p.is_dir():
        j = p / SURFACE_FILE
        return j if j.is_file() else None
    if p.is_file() and p.name == SURFACE_FILE:
        return p
    return None


def load_surface(path: str | Path) -> tuple[SurfaceRecord, Path]:
    """Load the artifact at *path* (directory or ``surface.json``) and its points file."""
    found = find_surface(path)
    if found is None:
        raise ContractError(f"{path}: not a surface artifact (no {SURFACE_FILE})")
    rec = SurfaceRecord.load(found)
    return rec, rec.points_path(found.parent)


def check_surface(rec: SurfaceRecord, points: Path, dataset_id: str, dataset_hash: str) -> None:
    """The §19 checks a claim-bearing evaluation runs before trusting an artifact.

    Identity, not quality. A surface built from another dataset — or from this one before its
    images changed — describes a different scene, and comparing it against this dataset's TLS
    reference would report a number about neither.
    """
    if rec.dataset_id != dataset_id:
        raise ContractError(
            f"surface {rec.surface_id} was built from dataset {rec.dataset_id!r}, "
            f"not {dataset_id!r}"
        )
    if rec.dataset_hash != dataset_hash:
        raise ContractError(
            f"surface {rec.surface_id} was built from dataset_hash {rec.dataset_hash[:12]}, "
            f"but {dataset_id} now hashes to {dataset_hash[:12]}; the dataset changed after the "
            "surface was made"
        )
    # frame/unit/point_count are Literal/gt-constrained on the model, so a record that parsed
    # cannot violate them. The file on disk can still be missing.
    if not points.is_file():
        raise ContractError(f"surface {rec.surface_id}: point_file {points} does not exist")
