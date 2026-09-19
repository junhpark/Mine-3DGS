"""The metric surface artifact (§1.7, Phase 1A).

A run's ``point_cloud/*.ply`` holds Gaussian *centres*: the parameters of a volumetric
radiance field, not samples of the tunnel wall. Comparing them against TLS measures how the
optimiser distributed its primitives, not how accurate the reconstruction is. So a surface has
to be *derived* — depth maps today, TSDF/mesh later — and the derivation has to leave a record.

``SurfaceRecord`` is that record, and nothing more. It says which run and which dataset the
samples came from, by what method, from what kind of depth, in what frame, and how many there
are. It states no accuracy.

Two things follow, and both are enforced rather than documented:

* **The record is not the surface.** ``check_surface`` re-reads the points and checks them
  against the digest the builder recorded. Otherwise "is this a surface?" degrades into "is
  there a file named surface.json next to it?", and a Gaussian PLY with a hand-written record
  walks straight back into a claim-bearing evaluation.
* **Being a surface is not enough for an accuracy claim.** Depth maps minegs did not render
  carry nothing that ties them to the run they are attributed to, so a surface built from them
  is structurally valid and diagnostically useful, and cannot support geometry_accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.pointcloud import PointCloud, read_ply
from minegs.core.provenance import ProvenanceRecord, sha256_file

SURFACE_FILE = "surface.json"
SURFACE_POINTS_FILE = "surface_points.ply"
SURFACE_DIRNAME = "surface"

#: Phase 1A ships one method. TSDF and mesh extraction are later phases (docs/ROADMAP.md);
#: adding them here before they exist would let a record name a provenance nothing can produce.
SurfaceMethod = Literal["depth_backprojection"]

#: Where the depth maps came from, which is what decides whether the surface can carry a
#: claim. ``external_unverified`` is everything Phase 1A can be handed: ``.npy`` files someone
#: put in a directory, with nothing linking them to the run the record names — the same maps
#: paired with any succeeded run would produce the same artifact. ``minegs_render`` is reserved
#: for the Phase 1B renderer, which will emit depth alongside the run id, dataset hash and a
#: per-map digest; the gate is written now so that path is turned on by producing the evidence,
#: not by remembering to add a check later.
DepthSource = Literal["external_unverified", "minegs_render"]

#: Depth sources whose evidence is strong enough for a geometry *accuracy* claim.
CLAIM_CAPABLE_DEPTH_SOURCES = frozenset({"minegs_render"})


class SurfaceRecord(VersionedModel):
    """``<surface_dir>/surface.json``."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    surface_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    method: SurfaceMethod
    depth_source: DepthSource
    # Not configurable. Surface samples are evaluated after a rigid hop into TLS_GLOBAL (§3),
    # which is only defined if they arrive in the dataset's metric training frame.
    frame: Literal["LOCAL_METRIC"] = "LOCAL_METRIC"
    unit: Literal["m"] = "m"
    #: Relative to the directory holding this file, so the artifact survives being moved.
    point_file: str = SURFACE_POINTS_FILE
    #: sha256 of ``point_file`` as the builder wrote it. What makes the record a statement
    #: about *these* points rather than about whatever now sits at that path.
    point_sha256: str = Field(min_length=1)
    point_count: int = Field(gt=0)
    depth_map_count: int = Field(gt=0)
    #: Order-independent digest over the depth maps consumed, so the fusion is reproducible
    #: and a later reader can tell two surfaces apart that differ only in their input.
    depth_sha256: str = Field(min_length=1)
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

    @property
    def supports_accuracy_claim(self) -> bool:
        """Whether this surface's depth provenance can carry ``geometry_accuracy`` (§1A)."""
        return self.depth_source in CLAIM_CAPABLE_DEPTH_SOURCES


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


def check_surface(
    rec: SurfaceRecord, points: Path, dataset_id: str, dataset_hash: str
) -> PointCloud:
    """Verify an artifact before an evaluation uses it, and return its points.

    Identity first: a surface built from another dataset — or from this one before its images
    changed — describes a different scene, and comparing it against this dataset's TLS
    reference would report a number about neither.

    Then the points themselves. A record is a claim *about* a file, so checking the record and
    trusting the file would make the boundary purely nominal: copy a Gaussian PLY, write a
    surface.json beside it, and "Gaussian centres are not surfaces" is enforced by nothing.
    The digest is what ties the two together; the count, frame and finiteness checks then say
    the record describes what the file actually contains.
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
    if not points.is_file():
        raise ContractError(f"surface {rec.surface_id}: point_file {points} does not exist")
    digest = sha256_file(points)
    if digest != rec.point_sha256:
        raise ContractError(
            f"surface {rec.surface_id}: {points} hashes to {digest[:12]}, but the record says "
            f"{rec.point_sha256[:12]}. These are not the points this surface was built from — "
            "rebuild the artifact with `minegs eval surface-depth` rather than editing it."
        )
    pc = read_ply(points)
    if len(pc) != rec.point_count:
        raise ContractError(
            f"surface {rec.surface_id}: {points} holds {len(pc)} points, record says "
            f"{rec.point_count}"
        )
    if pc.frame != rec.frame:
        raise ContractError(
            f"surface {rec.surface_id}: {points} is in frame {pc.frame}, record says {rec.frame}"
        )
    if not np.isfinite(pc.xyz).all():
        raise ContractError(f"surface {rec.surface_id}: {points} holds non-finite coordinates")
    return pc
