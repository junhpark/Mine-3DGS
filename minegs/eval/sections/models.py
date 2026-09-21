"""The section artifact (§11, Phase 1C) — sections with a provenance, not just numbers.

``SectionSeries`` is a geometry container: chainages, areas, radii. It says nothing about where
the points it was cut from came from, which is why ``minegs eval volume`` could be handed the
output of ``minegs eval sections raw/tls_full.ply`` and still reach ``volume_accuracy`` on the
strength of the dataset manifest alone. The manifest says the *dataset* can support the claim;
it cannot say anything about the cloud someone passed in.

``SectionRecord`` is the missing half. It answers, for one series:

    these sections are of which dataset, cut from which verified surface, from which run,
    from depth of which provenance, along which reference axis, with which parameters

and that is what the volume gate interrogates. The chain Phase 1A and 1B built —
``run → rendered depth → surface`` — ends here at ``→ sections → volume``.

Everything in the record is re-checked against the world before a claim uses it
(``check_section_record``): the dataset it names, the axis it was cut along, the station grid
that axis would produce today, and, when it is still on disk, the surface artifact itself. A
record is a statement about things that exist elsewhere, and an unchecked statement is a label.

Where those checks stop, stated plainly: they are about identity, and identity is not
arithmetic. They never open an ``area_m2``. Digests and the station grid tie a record to this
dataset, this axis and this surface, and that is all they do — so a record whose areas were
edited, or crafted to be self-consistent, passes every one of them. A boundary against
accident, not a signature: the same boundary ``DepthManifest`` draws, and the reason a
diagnostic number is allowed to be its producer's word.

A claim is held to more, and that work lives one module over. On the claim path
``check_claim_evidence`` (``minegs/eval/sections/build.py``) re-reads the surface this record
names, verifies its points against their own record, re-derives its Phase 1B promotion rather
than reading the recorded verdict, and cuts the sections again with the parameters this record
carries: the station count, which stations are observed, ``empty_bins``, ``n_points``, the
areas and the per-bin radii must all come back. The areas therefore *are* recomputed before a
``volume_accuracy`` claim — from the surface artifact, which is what a volume evaluation is
given, not from a cloud it was never handed — and a self-consistent forgery does not survive
it.

That costs a full re-extraction, so it happens on the claim path only, and it needs the surface
still on disk. When the surface is gone the claim is refused rather than taken on the record's
word, and a bare ``SectionSeries``, carrying no record at all, never reaches the claim path to
begin with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import ProvenanceRecord, sha256_file
from minegs.eval.sections.sections import SectionSeries
from minegs.eval.surface.models import CLAIM_CAPABLE_DEPTH_SOURCES

__all__ = [
    "SectionRecord",
    "SectionSource",
    "check_section_record",
    "load_section_input",
    "reference_axis_of",
]

#: Chainage slack: float representation only, never a tolerance on geometry (see
#: ``minegs/eval/volume/coverage.py``).
_EPS_M = 1e-9


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SectionSource(_Strict):
    """What the sections were cut from, in the terms that decide what they may support.

    ``raw_cloud`` is every PLY handed straight to ``minegs eval sections``: a TLS scan, a
    Gaussian centre dump, an export from another tool. Useful, and permanently diagnostic —
    nothing about the file says it samples the tunnel wall of *this* reconstruction.
    ``surface`` means a verified ``SurfaceRecord``, and then the depth provenance behind that
    surface decides the rest (§1A).
    """

    kind: Literal["surface", "raw_cloud"]
    #: Identity of the surface artifact, when there is one.
    surface_id: str | None = None
    run_id: str | None = None
    #: ``SurfaceRecord.depth_source``. ``minegs_render`` is the only claim-capable value.
    depth_source: str | None = None
    #: sha256 of the points that were sectioned — the surface's ``point_sha256``, or the raw
    #: PLY's own digest. Recorded for both so "which cloud was this?" always has an answer.
    point_sha256: str | None = None
    #: The path as the caller gave it. A convenience for a reader, and the handle that lets
    #: ``check_section_record`` re-verify the surface when it is still where it was.
    point_path: str | None = None

    @model_validator(mode="after")
    def _kind_carries_its_fields(self) -> SectionSource:
        """Every field a check reads must be there, or the check silently does not happen.

        ``run_id`` was compared against the surface only once it was known to be present;
        nulling it in the JSON would otherwise turn an equality into a no-op. The same argument
        holds for each of the others, so the shape is enforced rather than assumed.
        """
        surface_only = {
            "surface_id": self.surface_id,
            "run_id": self.run_id,
            "depth_source": self.depth_source,
        }
        if self.kind == "surface":
            blank = sorted(
                k
                for k, v in {
                    **surface_only,
                    "point_sha256": self.point_sha256,
                    "point_path": self.point_path,
                }.items()
                if not v
            )
            if blank:
                raise ValueError(f"a surface-backed section source must name {blank}")
        else:
            named = sorted(k for k, v in surface_only.items() if v is not None)
            if named:
                raise ValueError(f"a {self.kind} section source cannot name {named}")
            if not self.point_sha256 or not self.point_path:
                raise ValueError("a section source must name the cloud it was cut from")
        return self

    @property
    def supports_accuracy_claim(self) -> bool:
        """Whether sections cut from this source can carry ``volume_accuracy`` (§1C)."""
        return self.kind == "surface" and self.depth_source in CLAIM_CAPABLE_DEPTH_SOURCES


class SectionRecord(VersionedModel):
    """A section series plus the provenance a volume claim has to stand on."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    section_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    source: SectionSource
    #: ``centerline:<source>:<file>`` — the same string ``VolumeReport.reference_axis`` carries.
    reference_axis: str = Field(min_length=1)
    #: sha256 of the axis file itself. ``dataset_hash`` already covers ``centerline.csv`` at the
    #: dataset root, but the manifest may name another file, and chainage is meaningless against
    #: a different polyline: an axis 40 cm to the left renumbers every station.
    reference_axis_sha256: str = Field(min_length=1)
    # Sections are cut in TLS_GLOBAL (§3): that is the frame the reference axis, the holdout
    # ranges and the TLS reference all live in.
    frame: Literal["TLS_GLOBAL"] = "TLS_GLOBAL"
    unit: Literal["m"] = "m"
    series: SectionSeries
    parameters: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceRecord

    @property
    def supports_accuracy_claim(self) -> bool:
        return self.source.supports_accuracy_claim


def reference_axis_of(manifest) -> str:
    """The axis string for a dataset manifest, in one place so writer and checker agree."""
    c = manifest.centerline
    if c is None:
        raise ContractError(
            f"dataset {manifest.dataset_id} has no centerline; sections and volume are defined "
            "along a reference axis (§10)"
        )
    return f"centerline:{c.source}:{c.file}"


def load_section_input(path: str | Path) -> tuple[SectionRecord | None, SectionSeries]:
    """Read a sections JSON as either a record or a bare legacy series.

    Returning ``None`` for the bare series rather than refusing keeps every diagnostic use of a
    hand-made or pre-1C series working — ``eval change``, a plot, a spreadsheet. What the
    ``None`` costs is the claim: a series that carries no provenance has nothing for the volume
    gate to check, so it can never be more than diagnostic.
    """
    from minegs.core.config import load_structured

    path = Path(path)
    if not path.is_file():
        raise ContractError(f"{path}: no such sections file")
    raw = load_structured(path)
    if not isinstance(raw, dict):
        raise ContractError(f"{path}: expected a mapping at top level")
    if "section_id" in raw or "series" in raw:
        rec = SectionRecord.load(path)
        return rec, rec.series
    try:
        return None, SectionSeries.model_validate(raw)
    except Exception as e:
        raise ContractError(
            f"{path}: not a section artifact (no 'section_id') and not a bare section series "
            f"({e}). Produce one with `minegs eval sections`."
        ) from e


def check_section_record(
    rec: SectionRecord,
    dataset_id: str,
    dataset_hash: str,
    dataset_dir: Path,
    manifest,
    centerline,
) -> None:
    """Everything the record asserts, checked against the dataset as it is now.

    Called on every path, diagnostic included. A series whose *provenance* points at another
    dataset is not a weaker number, it is the wrong tunnel — the same reason ``check_surface``
    runs before a diagnostic geometry comparison.
    """
    if rec.dataset_id != dataset_id:
        raise ContractError(
            f"sections {rec.section_id} were cut from dataset {rec.dataset_id!r}, "
            f"not {dataset_id!r}"
        )
    if rec.dataset_hash != dataset_hash:
        raise ContractError(
            f"sections {rec.section_id} were cut from dataset_hash {rec.dataset_hash[:12]}, but "
            f"{dataset_id} now hashes to {dataset_hash[:12]}; the dataset — its centerline, its "
            "cameras or its images — changed after the sections were cut"
        )
    if rec.series.frame != rec.frame:
        raise ContractError(
            f"sections {rec.section_id}: the record declares frame {rec.frame}, the series says "
            f"{rec.series.frame}"
        )
    _check_parameters(rec)
    _check_axis(rec, dataset_dir, manifest)
    _check_station_grid(rec, centerline)
    _check_surface_still_agrees(rec, dataset_id, dataset_hash)


def _check_parameters(rec: SectionRecord) -> None:
    """The parameters block and the series must describe the same cut.

    The grid below is re-derived from ``parameters``, so a record whose ``interval_m`` says one
    thing and whose series says another would be checked against the wrong grid — and the two
    are written from the same call, so disagreeing means one of them was edited.
    """
    for key, got in (
        ("interval_m", rec.series.interval_m),
        ("thickness_m", rec.series.thickness_m),
        ("angle_bins", rec.series.angle_bins),
    ):
        want = rec.parameters.get(key)
        if want is not None and want != got:
            raise ContractError(
                f"sections {rec.section_id}: parameters say {key}={want!r}, the series says "
                f"{got!r}; one of the two was edited after the sections were cut"
            )


def _check_axis(rec: SectionRecord, dataset_dir: Path, manifest) -> None:
    axis = reference_axis_of(manifest)
    if rec.reference_axis != axis:
        raise ContractError(
            f"sections {rec.section_id} were cut along {rec.reference_axis!r}, but this dataset's "
            f"axis is {axis!r}; chainage means something different on each"
        )
    path = dataset_dir / manifest.centerline.file
    if not path.is_file():
        raise ContractError(f"{path}: the dataset's reference axis file is missing")
    digest = sha256_file(path)
    if digest != rec.reference_axis_sha256:
        raise ContractError(
            f"sections {rec.section_id}: {manifest.centerline.file} hashes to {digest[:12]}, but "
            f"the sections were cut against {rec.reference_axis_sha256[:12]}. The axis was "
            "edited, so every chainage in this series refers to a polyline that no longer exists."
        )


def _check_station_grid(rec: SectionRecord, centerline) -> None:
    """The recorded chainages must be the grid this axis and these parameters produce today.

    The strongest thing available without the point cloud. ``extract_sections`` takes its
    stations from ``Centerline.stations(interval, start, end)``, so the grid is reproducible
    from the record's own parameters — and it is a property of the *axis*, not of the record, so
    a series cut along a longer tunnel, at another interval, or over another sub-range does not
    survive being re-derived here.
    """
    p = rec.parameters
    try:
        expected = centerline.stations(
            float(rec.series.interval_m), p.get("start_m"), p.get("end_m")
        )
    except (TypeError, ValueError) as e:
        raise ContractError(f"sections {rec.section_id}: unusable station parameters ({e})") from e
    got = np.asarray(rec.series.chainages(), dtype=np.float64)
    if len(got) != len(expected) or (
        len(got) and float(np.max(np.abs(np.sort(got) - expected))) > _EPS_M
    ):
        raise ContractError(
            f"sections {rec.section_id}: the series has {len(got)} stations, but this dataset's "
            f"reference axis at interval {rec.series.interval_m} m over "
            f"[{p.get('start_m')}, {p.get('end_m')}] has {len(expected)}. These sections were not "
            "cut along the axis this dataset now declares."
        )


def _check_surface_still_agrees(rec: SectionRecord, dataset_id: str, dataset_hash: str) -> None:
    """Re-verify the source surface when it is still on disk.

    The record names a surface it no longer holds, exactly as a depth manifest names a
    checkpoint. When the artifact is still there the cheap thing is to check it rather than
    believe the copy — a surface rebuilt in place, or demoted after its depth manifest stopped
    verifying, would otherwise keep exporting a claim through a section record written earlier.
    A surface that has been archived or deleted leaves the recorded ids to agree on, which is
    the same bargain Phase 1B struck for a deleted checkpoint.
    """
    from minegs.eval.surface.models import check_surface, find_surface, load_surface

    if rec.source.kind != "surface" or not rec.source.point_path:
        return
    path = Path(rec.source.point_path)
    if find_surface(path) is None:
        return
    surface, points = load_surface(path)
    check_surface(surface, points, dataset_id, dataset_hash)
    if surface.surface_id != rec.source.surface_id:
        raise ContractError(
            f"sections {rec.section_id} were cut from surface {rec.source.surface_id}, but "
            f"{path} now holds {surface.surface_id}"
        )
    # The run is the far end of the chain this whole phase exists to keep unbroken, and it was
    # recorded without ever being compared: editing one line of sections.json attributed a
    # volume measured on one run to another, with every other check still passing and the
    # forged id travelling into the published report.
    if surface.run_id != rec.source.run_id:
        raise ContractError(
            f"sections {rec.section_id} record run {rec.source.run_id!r}, but surface "
            f"{surface.surface_id} belongs to run {surface.run_id!r}"
        )
    if surface.point_sha256 != rec.source.point_sha256:
        raise ContractError(
            f"sections {rec.section_id}: {path} now holds points hashing to "
            f"{surface.point_sha256[:12]}, the sections were cut from "
            f"{str(rec.source.point_sha256)[:12]}"
        )
    # Directional on purpose. The section builder records the *re-derived* provenance, not the
    # string surface.json carries, so a record saying less than the file is the honest case and
    # must keep working. A record saying *more* than the file is the one that would promote.
    if (
        rec.source.depth_source in CLAIM_CAPABLE_DEPTH_SOURCES
        and surface.depth_source not in CLAIM_CAPABLE_DEPTH_SOURCES
    ):
        raise ContractError(
            f"sections {rec.section_id} record depth_source={rec.source.depth_source!r}, but "
            f"surface {surface.surface_id} is {surface.depth_source!r}"
        )
