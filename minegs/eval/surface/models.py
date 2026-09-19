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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


# ---------------------------------------------------------------- Phase 1B: rendered depth

DEPTH_MANIFEST_FILE = "depth_manifest.json"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RenderedDepth(_Strict):
    """One depth map the renderer produced, and enough to recognise it again."""

    image_id: int
    camera_id: int
    image_name: str
    #: Relative to the directory holding the manifest.
    file: str
    width: int
    height: int
    sha256: str = Field(min_length=1)
    #: Fraction of pixels carrying a usable range. The rest are NaN by policy, not by accident:
    #: a ray that hit nothing has no depth, and writing 0 there would back-project to the camera
    #: centre and read as a surface.
    valid_ratio: float = Field(ge=0.0, le=1.0)
    min_m: float | None = None
    max_m: float | None = None

    @field_validator("file")
    @classmethod
    def _relative_name(cls, v: str) -> str:
        p = Path(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"file must be relative to the manifest directory, got {v!r}")
        return v


class DepthManifest(VersionedModel):
    """``<depth_dir>/depth_manifest.json`` — what makes rendered depth *evidence* (Phase 1B).

    Phase 1A can be handed a directory of ``.npy`` files and has no way to tell whether they
    came from the run they are attributed to, so the surfaces it builds are diagnostic-only.
    This manifest is the difference: it is written by minegs' own renderer, in the same pass
    that produced the maps, and it names the run, the dataset, the checkpoint and a digest per
    map. A surface built from depth whose manifest still verifies against all four is the one
    thing allowed to claim ``depth_source="minegs_render"``.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    manifest_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    #: ``{"name": ..., "version": ...}`` of the training backend whose weights were rendered.
    backend: dict[str, str]
    #: ``{"file": run-relative path, "sha256": ..., "step": ...}``. Identity of the weights,
    #: so two manifests from two checkpoints of the same run are not interchangeable.
    checkpoint: dict[str, Any]
    #: ``{"name": ..., "version": ..., "settings": {...}}`` — who rendered, and under what knobs.
    renderer: dict[str, Any]
    frame: Literal["LOCAL_METRIC"] = "LOCAL_METRIC"
    unit: Literal["m"] = "m"
    depths: list[RenderedDepth] = Field(min_length=1)
    #: What the run was actually trained on, copied from ``run.json``'s ``staged``. Depth is
    #: rendered for every dataset view, but a profile with ``max_images`` trains on a subset, so
    #: some of those views were never supervised. That does not make the render wrong — it makes
    #: two otherwise identical artifacts distinguishable, which they were not before.
    staged: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceRecord

    def by_image_name(self) -> dict[str, RenderedDepth]:
        return {d.image_name: d for d in self.depths}


def find_depth_manifest(depth_dir: str | Path) -> Path | None:
    """``depth_manifest.json`` in *depth_dir*, or ``None`` for a plain directory of maps."""
    p = Path(depth_dir) / DEPTH_MANIFEST_FILE
    return p if p.is_file() else None


def verify_depth_manifest(
    manifest: DepthManifest,
    depth_dir: Path,
    run,
    run_dir: Path,
    dataset_id: str,
    dataset_hash: str,
    cameras: dict,
    images: dict,
) -> None:
    """Everything that has to hold before rendered depth counts as evidence (Phase 1B §5).

    Presence of a manifest proves nothing on its own — that was the whole objection to a
    ``depth_source`` string. What makes it evidence is that it still agrees with the run it
    names, the dataset as it is now, the camera model being back-projected against, and the
    bytes on disk. Any one of those drifting means the maps are no longer describing the thing
    the surface will be compared to, so this refuses rather than downgrading silently.
    """
    from minegs.eval.surface.depth import depth_map_path

    where = depth_dir / DEPTH_MANIFEST_FILE
    if manifest.run_id != run.run_id:
        raise ContractError(
            f"{where}: depth was rendered from run {manifest.run_id!r}, but --run-dir names "
            f"{run.run_id!r}"
        )
    if manifest.dataset_id != dataset_id:
        raise ContractError(
            f"{where}: depth was rendered against dataset {manifest.dataset_id!r}, "
            f"not {dataset_id!r}"
        )
    if manifest.dataset_hash != dataset_hash:
        raise ContractError(
            f"{where}: depth was rendered against dataset_hash {manifest.dataset_hash[:12]}, "
            f"but {dataset_id} now hashes to {dataset_hash[:12]}; the cameras these maps were "
            "rendered for are not the cameras they would be back-projected with"
        )

    _verify_checkpoint_identity(manifest, where, run, run_dir)
    _verify_renderer_and_staging(manifest, where, run)

    entries = manifest.by_image_name()
    if len(entries) != len(manifest.depths):
        raise ContractError(f"{where}: two entries name the same image")
    expected = {im.name for im in images.values()}
    missing = sorted(expected - set(entries))
    extra = sorted(set(entries) - expected)
    if missing or extra:
        raise ContractError(
            f"{where}: the manifest covers {len(entries)} views, the dataset has "
            f"{len(expected)}"
            + (f"; missing {missing[:6]}" if missing else "")
            + (f"; unknown {extra[:6]}" if extra else "")
        )

    by_name = {im.name: im for im in images.values()}
    for name, entry in sorted(entries.items()):
        im = by_name[name]
        cam = cameras[im.camera_id]
        if entry.camera_id != im.camera_id or entry.image_id != im.id:
            raise ContractError(
                f"{where}: {name} is recorded against camera {entry.camera_id} / image "
                f"{entry.image_id}, but the dataset has camera {im.camera_id} / image {im.id}"
            )
        if (entry.width, entry.height) != (cam.width, cam.height):
            raise ContractError(
                f"{where}: {name} was rendered at {entry.width}x{entry.height}, but camera "
                f"{cam.id} is {cam.width}x{cam.height}"
            )
        # The fuser finds its maps by the naming contract, not by reading this field, so a
        # manifest naming some *other* file would have its digest checked while a different
        # file was consumed. Pinning the two together is what keeps "verified bytes" and
        # "back-projected bytes" the same bytes.
        expected_file = depth_map_path(depth_dir, name).name
        if entry.file != expected_file:
            raise ContractError(
                f"{where}: {name} is recorded as {entry.file!r}, but the depth naming contract "
                f"makes it {expected_file!r}. The map that would be verified is not the map that "
                "would be back-projected."
            )
        f = depth_dir / entry.file
        if not f.is_file():
            raise ContractError(f"{where}: {entry.file} is listed but missing")
        digest = sha256_file(f)
        if digest != entry.sha256:
            raise ContractError(
                f"{where}: {entry.file} hashes to {digest[:12]}, the manifest says "
                f"{entry.sha256[:12]}. This depth map was changed after it was rendered."
            )


def _verify_checkpoint_identity(manifest: DepthManifest, where: Path, run, run_dir: Path) -> None:
    """The weights named by the manifest must be the weights the run ended on.

    Recording the checkpoint was only half of it: a manifest rendered from an earlier
    checkpoint of the same run passes every other check — same run id, same dataset, same
    views, same digests — while describing a different model than the one `run.json` presents
    as the result. The digest is re-checked when the file is still there, which is the case
    that matters; a checkpoint deleted after rendering leaves the file and step to agree on.
    """
    recorded = manifest.checkpoint or {}
    file, step = recorded.get("file"), recorded.get("step")
    if file != run.final_checkpoint:
        raise ContractError(
            f"{where}: depth was rendered from checkpoint {file!r}, but run {run.run_id} ended "
            f"on {run.final_checkpoint!r}"
        )
    if step != run.checkpoint_step:
        raise ContractError(
            f"{where}: depth was rendered at step {step}, but run {run.run_id} ended at "
            f"{run.checkpoint_step}"
        )
    if not file:
        raise ContractError(f"{where}: the manifest names no checkpoint")
    ckpt = Path(run_dir) / file
    if ckpt.is_file():
        digest = sha256_file(ckpt)
        if digest != recorded.get("sha256"):
            raise ContractError(
                f"{where}: {file} hashes to {digest[:12]}, but the depth was rendered from "
                f"{str(recorded.get('sha256'))[:12]}. These are different weights."
            )


def _verify_renderer_and_staging(manifest: DepthManifest, where: Path, run) -> None:
    """The manifest's own account of who rendered and what was trained, checked against reality.

    Recording a field and never comparing it is how ``checkpoint`` slipped through review, so
    the two added since get the same treatment. ``renderer.name`` must be a renderer this build
    can produce — otherwise the manifest was minted through the injection seam that exists for
    testing, and says nothing about minegs having rendered anything. ``staged`` must still match
    the run, so a manifest cannot describe a full-dataset run while the record says a subset.
    """
    from minegs.eval.surface.render import known_renderer_names

    name = (manifest.renderer or {}).get("name")
    known = known_renderer_names()
    if name not in known:
        raise ContractError(
            f"{where}: depth was produced by renderer {name!r}, which this build does not ship "
            f"(known: {sorted(known)}). A manifest from an unknown renderer is not evidence "
            "that minegs rendered the depth."
        )
    recorded = manifest.staged or {}
    actual = {k: v for k, v in (run.staged or {}).items() if k in recorded}
    if recorded and actual != recorded:
        raise ContractError(
            f"{where}: the manifest records staging {recorded}, but run {run.run_id} now "
            f"records {actual}; the depth was rendered for a different staging of this run"
        )
