"""Station/scan <-> image mapping (Phase 0B.2) — evidence only, never inference.

The question this module answers is "which scan does this image belong to?", and the only
acceptable answers are ones the *data* states. Three evidence tiers are recognised:

======================= ================================ ==========
evidence_type           where it comes from              status
======================= ================================ ==========
``e57_associated_guid`` the E57's own ``associatedData3DGuid``  ``confirmed``
``vendor_manifest``     a machine-generated vendor index ``confirmed``
``explicit_mapping``    a mapping file the user wrote    ``manual``
======================= ================================ ==========

Everything else is refused. In particular none of the following ever produces a mapping, no
matter how convincing it looks on a given vendor's export (§10):

* scan index == image index, or equal scan and image counts;
* file name order, lexical sort, or any fuzzy name similarity;
* timestamp/EXIF proximity;
* an undocumented vendor naming convention;
* the nearest scanner pose, absent validated camera geometry.

Such observations may be printed as *hints* — a hint tells the user where to look when they
write a mapping file; it never becomes a ``MappingRecord`` with a target. The cost of being
wrong here is silent: a panorama attributed to the wrong station produces a reconstruction
that trains, converges and is geometrically meaningless.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.provenance import (
    ProvenanceRecord,
    SourceAsset,
    git_commit,
    sha256_file,
    tool_versions,
)
from minegs.ingest.e57.images import ImageAsset, discover_embedded_images, discover_external_images
from minegs.ingest.e57.models import E57Inventory

#: What justified a mapping. ``none`` means nothing did, which is a legitimate answer.
EvidenceType = Literal["e57_associated_guid", "vendor_manifest", "explicit_mapping", "none"]

#: ``confirmed`` - unique machine-verifiable evidence
#: ``manual``    - the user said so
#: ``unmapped``  - no evidence at all
#: ``ambiguous`` - one piece of evidence points at several targets
#: ``orphan``    - evidence exists but names a target this file does not contain
#: ``conflict``  - two pieces of evidence disagree
MappingStatus = Literal["confirmed", "manual", "unmapped", "ambiguous", "orphan", "conflict"]

#: How a mapping file refers to an image and to its target.
ImageRefKind = Literal["image_id", "image_name", "image_guid"]
TargetKind = Literal["scan_id", "scan_guid", "station_id"]

_IMAGE_REF_COLUMNS: dict[str, ImageRefKind] = {
    "image_id": "image_id",
    "image_name": "image_name",
    "image_guid": "image_guid",
}
_TARGET_COLUMNS: dict[str, TargetKind] = {
    "scan_id": "scan_id",
    "scan_guid": "scan_guid",
    "station_id": "station_id",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetRef(_Strict):
    """A mapping target, named the way the mapping file named it."""

    kind: TargetKind
    value: str

    def __str__(self) -> str:
        return f"{self.kind}={self.value}"


class ExplicitEntry(_Strict):
    """One row of a user mapping file or a vendor manifest."""

    image_ref: str
    image_ref_kind: ImageRefKind
    target: TargetRef
    #: ``<file>:<row>`` — so a bad row can be found and fixed.
    origin: str
    evidence_type: Literal["explicit_mapping", "vendor_manifest"] = "explicit_mapping"


class UnresolvedReference(_Strict):
    """A mapping row naming something this survey does not contain.

    Kept as its own list rather than folded into ``mappings``: a row referring to
    ``image_042`` in a file with eight images has no ``image_id`` to be a record of, and
    dropping it silently would let a typo in a mapping file look like a clean run.
    """

    origin: str
    image_ref: str
    image_ref_kind: ImageRefKind
    target: TargetRef
    reason: str


class MappingRecord(_Strict):
    """What we know about one image's station, and why we know it."""

    image_id: str
    #: Filled only for ``confirmed`` and ``manual``. Every other status means "we do not know",
    #: and writing a plausible guess here is the failure mode this module exists to prevent.
    scan_id: str | None = None
    #: The inferred one-per-scan station candidate for ``scan_id`` (Phase 0B.1), or ``None``.
    station_id: str | None = None
    status: MappingStatus
    evidence_type: EvidenceType
    evidence_value: str | None = None
    reason: str
    #: For ``ambiguous``: every target the evidence could mean. Never resolved by first match.
    candidate_scan_ids: list[str] = Field(default_factory=list)
    #: Observations that may help a human write a mapping file. Never evidence (§10).
    hints: list[str] = Field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.status in ("confirmed", "manual") and self.scan_id is not None


class PanoMappingReport(VersionedModel):
    """``minegs ingest e57 pano-map`` result.

    A separate versioned artifact rather than more fields on ``E57Inventory``: the inventory
    answers "what is in this file", this answers "what belongs with what", and the second
    question can be re-answered (with a new mapping file) without re-reading the first.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    source_file: str
    source_sha256: str | None = None
    hash_skipped_reason: str | None = None
    #: Set when the images came from a directory rather than from the E57 itself.
    image_root: str | None = None
    scan_count: int = 0

    images: list[ImageAsset] = Field(default_factory=list)
    mappings: list[MappingRecord] = Field(default_factory=list)
    unresolved_references: list[UnresolvedReference] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    def record_for(self, image_id: str) -> MappingRecord:
        for m in self.mappings:
            if m.image_id == image_id:
                return m
        raise KeyError(image_id)

    def by_status(self, status: MappingStatus) -> list[MappingRecord]:
        return [m for m in self.mappings if m.status == status]

    def resolved(self) -> list[MappingRecord]:
        return [m for m in self.mappings if m.is_resolved]

    def has_any_issue(self) -> bool:
        return bool(self.issues) or bool(self.unresolved_references)


# ---------------------------------------------------------------- mapping file readers


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ContractError(f"{path}: mapping file is empty")
        names = [(n or "").strip() for n in reader.fieldnames]
        if not any(n in _IMAGE_REF_COLUMNS for n in names) or not any(
            n in _TARGET_COLUMNS for n in names
        ):
            raise ContractError(
                f"{path}: header {names} does not name an image column "
                f"({sorted(_IMAGE_REF_COLUMNS)}) and a target column ({sorted(_TARGET_COLUMNS)}). "
                "Column order does not matter; column names do, because a two-column file with "
                "no header cannot say which side is which."
            )
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def _rows_from_json(data: Any, path: Path) -> list[dict[str, str]]:
    if isinstance(data, dict):
        for key in ("mappings", "images", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ContractError(
                f"{path}: JSON object has no 'mappings' list. A bare "
                "{image: target} object is ambiguous about what the target names."
            )
    if not isinstance(data, list):
        raise ContractError(f"{path}: expected a list of mapping records")
    out = []
    for row in data:
        if not isinstance(row, dict):
            raise ContractError(f"{path}: every mapping record must be an object, got {type(row)}")
        out.append({str(k).strip(): str(v).strip() for k, v in row.items() if v is not None})
    return out


def _entries_from_rows(
    rows: list[dict[str, str]],
    path: Path,
    evidence_type: Literal["explicit_mapping", "vendor_manifest"],
    row_label: str = "line",
) -> list[ExplicitEntry]:
    """Rows -> entries, refusing any row that does not name exactly one image and one target.

    ``row_label`` only shapes the ``origin`` string: a CSV row is a line number the user can
    jump to, a JSON row is the n-th record.
    """
    entries: list[ExplicitEntry] = []
    # A CSV's first line is the header, so the first data row is line 2; JSON records count
    # from 1. Either way the origin points at something the user can actually find.
    start = 2 if row_label == "line" else 1
    for i, row in enumerate(rows, start=start):
        image_cols = [(c, k) for c, k in _IMAGE_REF_COLUMNS.items() if row.get(c)]
        target_cols = [(c, k) for c, k in _TARGET_COLUMNS.items() if row.get(c)]
        where = f"{path.name}:{row_label} {i}"
        if not image_cols or not target_cols:
            raise ContractError(
                f"{where}: row {row} names "
                + ("no image" if not image_cols else "no target")
                + "; every row must name both, or the mapping is a guess"
            )
        if len(image_cols) > 1 or len(target_cols) > 1:
            raise ContractError(
                f"{where}: row {row} names more than one image or target column. Pick one "
                "identifier per side so the row means exactly one thing."
            )
        (image_col, image_kind), (target_col, target_kind) = image_cols[0], target_cols[0]
        entries.append(
            ExplicitEntry(
                image_ref=row[image_col],
                image_ref_kind=image_kind,
                target=TargetRef(kind=target_kind, value=row[target_col]),
                origin=where,
                evidence_type=evidence_type,
            )
        )
    if not entries:
        raise ContractError(f"{path}: mapping file contains no rows")
    return entries


def read_mapping_file(path: str | Path) -> list[ExplicitEntry]:
    """Read a user-written mapping (Tier B). Produces ``manual`` mappings.

    CSV with a header, or JSON. One image column (``image_id`` / ``image_name`` /
    ``image_guid``) and one target column (``scan_id`` / ``scan_guid`` / ``station_id``)::

        scan_id,image_id
        scan_000,image_002
        scan_001,image_004
    """
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"mapping file not found: {p}")
    if p.suffix.lower() == ".json":
        return _entries_from_rows(
            _rows_from_json(json.loads(p.read_text()), p), p, "explicit_mapping", "record"
        )
    return _entries_from_rows(_rows_from_csv(p), p, "explicit_mapping")


def read_vendor_manifest(path: str | Path) -> list[ExplicitEntry]:
    """Read a machine-generated vendor index (Tier C). Produces ``confirmed`` mappings.

    The difference from :func:`read_mapping_file` is not the columns — it is the claim. A
    vendor manifest is written by the capture software and is therefore machine-verifiable;
    a mapping a person typed is not, however careful they were. So this reader refuses
    anything that does not identify itself as a vendor export::

        {"vendor": "Matterport", "generated_by": "...", "mappings": [
            {"image_id": "image_000", "scan_guid": "{...}"}
        ]}

    Without that guard "Tier C" would just be "Tier B with a better status", and a hand-typed
    file would be reported to the reviewer as confirmed evidence.
    """
    p = Path(path)
    if not p.is_file():
        raise ContractError(f"vendor manifest not found: {p}")
    if p.suffix.lower() != ".json":
        raise ContractError(
            f"{p}: a vendor manifest must be JSON declaring 'vendor' and 'generated_by'. "
            "A CSV cannot state who produced it — pass it with --mapping instead, which "
            "records the mapping as manual rather than confirmed."
        )
    data = json.loads(p.read_text())
    if not isinstance(data, dict) or not data.get("vendor") or not data.get("generated_by"):
        raise ContractError(
            f"{p}: a vendor manifest must declare both 'vendor' and 'generated_by' at the top "
            "level. Only a machine-generated index counts as confirmed evidence; pass a "
            "hand-written mapping with --mapping, which records it as manual."
        )
    entries = _entries_from_rows(_rows_from_json(data, p), p, "vendor_manifest", "record")
    vendor = str(data["vendor"])
    return [e.model_copy(update={"origin": f"{vendor}:{e.origin}"}) for e in entries]


# ---------------------------------------------------------------- the mapping itself


def _normalise_guid(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _loose_guid(value: str) -> str:
    """A GUID reduced to the part vendors agree on, for *hints* only."""
    return value.strip().strip("{}").replace("-", "").lower()


class _Index:
    """Lookups over one inventory. Every one is exact; near matches become hints."""

    def __init__(self, inv: E57Inventory) -> None:
        self.inv = inv
        self.scan_ids = {s.scan_id for s in inv.scans}
        self.guid_to_scans: dict[str, list[str]] = defaultdict(list)
        self.loose_guid_to_scans: dict[str, list[str]] = defaultdict(list)
        for s in inv.scans:
            g = _normalise_guid(s.guid)
            if g is not None:
                self.guid_to_scans[g].append(s.scan_id)
                self.loose_guid_to_scans[_loose_guid(g)].append(s.scan_id)
        self.station_to_scans: dict[str, list[str]] = {
            c.station_id: list(c.scan_ids) for c in inv.station_candidates
        }
        self.scan_to_station: dict[str, str] = {}
        for c in inv.station_candidates:
            for sid in c.scan_ids:
                self.scan_to_station.setdefault(sid, c.station_id)

    def resolve(self, target: TargetRef) -> tuple[list[str], str | None]:
        """Target -> (matching scan_ids, hint). An empty list means the target does not exist."""
        if target.kind == "scan_id":
            return ([target.value] if target.value in self.scan_ids else []), None
        if target.kind == "station_id":
            return list(self.station_to_scans.get(target.value, [])), None
        matches = list(self.guid_to_scans.get(_normalise_guid(target.value) or "", []))
        if matches:
            return matches, None
        near = self.loose_guid_to_scans.get(_loose_guid(target.value), [])
        hint = (
            f"no scan GUID equals {target.value!r}, but {near} differ only in case, braces or "
            "hyphens; that is a vendor spelling difference this reader will not resolve for "
            "you — confirm it and write an explicit scan_id mapping"
            if near
            else None
        )
        return [], hint


def _resolve_image_ref(entry: ExplicitEntry, images: list[ImageAsset]) -> list[ImageAsset]:
    if entry.image_ref_kind == "image_id":
        return [im for im in images if im.image_id == entry.image_ref]
    if entry.image_ref_kind == "image_name":
        return [im for im in images if (im.name or "") == entry.image_ref]
    return [im for im in images if _normalise_guid(im.guid) == _normalise_guid(entry.image_ref)]


def _embedded_evidence(
    image: ImageAsset, index: _Index
) -> tuple[MappingStatus, list[str], str | None, str | None]:
    """(status, candidate scan_ids, evidence_value, hint) from the image's own association."""
    guid = _normalise_guid(image.associated_scan_guid)
    if guid is None:
        return "unmapped", [], None, None
    matches, hint = index.resolve(TargetRef(kind="scan_guid", value=guid))
    if len(matches) == 1:
        return "confirmed", matches, guid, None
    if len(matches) > 1:
        return "ambiguous", matches, guid, None
    return "orphan", [], guid, hint


def map_images(
    inv: E57Inventory,
    images: list[ImageAsset],
    entries: list[ExplicitEntry] | None = None,
) -> tuple[list[MappingRecord], list[UnresolvedReference], list[str], list[str]]:
    """Map images to scans. Returns (records, unresolved references, issues, notes).

    One record per image, always — an image with no evidence is reported as ``unmapped``
    rather than omitted, because "we looked and found nothing" and "we did not look" must not
    produce the same report.
    """
    index = _Index(inv)
    entries = list(entries or [])
    issues: list[str] = []
    notes: list[str] = []

    # ---- resolve every mapping row to concrete images, keeping the rows that resolve nowhere
    unresolved: list[UnresolvedReference] = []
    by_image: dict[str, list[tuple[ExplicitEntry, list[str], str | None]]] = defaultdict(list)
    for entry in entries:
        targets = _resolve_image_ref(entry, images)
        if not targets:
            unresolved.append(
                UnresolvedReference(
                    origin=entry.origin,
                    image_ref=entry.image_ref,
                    image_ref_kind=entry.image_ref_kind,
                    target=entry.target,
                    reason=f"no discovered image has {entry.image_ref_kind} "
                    f"{entry.image_ref!r}; the mapping row applies to nothing",
                )
            )
            continue
        if len(targets) > 1:
            unresolved.append(
                UnresolvedReference(
                    origin=entry.origin,
                    image_ref=entry.image_ref,
                    image_ref_kind=entry.image_ref_kind,
                    target=entry.target,
                    reason=f"{entry.image_ref_kind} {entry.image_ref!r} matches "
                    f"{[t.image_id for t in targets]}; refer to one image by image_id instead",
                )
            )
            continue
        scan_ids, hint = index.resolve(entry.target)
        by_image[targets[0].image_id].append((entry, scan_ids, hint))

    records = [
        _record_for(im, index, by_image.get(im.image_id, []), bool(entries)) for im in images
    ]

    # ---- report-level findings
    for ref in unresolved:
        issues.append(f"{ref.origin}: {ref.reason}")
    dup_scan_guids = {g: ids for g, ids in index.guid_to_scans.items() if len(ids) > 1}
    for guid, ids in dup_scan_guids.items():
        issues.append(
            f"scan GUID {guid} is declared by {len(ids)} scans ({ids}); any image associated "
            "with it is ambiguous and cannot be mapped automatically"
        )
    image_guids: dict[str, list[str]] = defaultdict(list)
    for im in images:
        g = _normalise_guid(im.guid)
        if g is not None:
            image_guids[g].append(im.image_id)
    for guid, ids in image_guids.items():
        if len(ids) > 1:
            issues.append(f"image GUID {guid} is declared by {len(ids)} images ({ids})")

    claimed: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if rec.scan_id is not None:
            claimed[rec.scan_id].append(rec.image_id)
    for scan_id, image_ids in claimed.items():
        if len(image_ids) > 1:
            notes.append(
                f"{len(image_ids)} images map to {scan_id} ({image_ids}); that is legitimate "
                "for a multi-image station, but check it is what you meant"
            )

    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.status] += 1
    if images:
        notes.append(
            "mapping status: "
            + ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
            + f" (of {len(images)} images, {inv.scan_count} scans)"
        )
    if counts["unmapped"] == len(images) and images:
        notes.append(
            "no image carries mapping evidence. Equal image and scan counts, image order and "
            "file names are NOT evidence (§10) — write a mapping file and pass --mapping"
        )
    return records, unresolved, issues, notes


def _record_for(
    image: ImageAsset,
    index: _Index,
    rows: list[tuple[ExplicitEntry, list[str], str | None]],
    mapping_given: bool = False,
) -> MappingRecord:
    """One image's mapping, combining its own declaration with any mapping-file rows."""
    emb_status, emb_scans, emb_value, emb_hint = _embedded_evidence(image, index)
    hints = [h for h in (emb_hint,) if h]

    if not rows:
        reason = {
            "confirmed": "the E57 associates this image with exactly one scan GUID",
            "ambiguous": "the associated scan GUID is declared by more than one scan, so the "
            "association does not identify a single scan; first match is not a resolution",
            "orphan": "the associated scan GUID names a scan this file does not contain",
            "unmapped": "the image declares no associatedData3DGuid and "
            + ("no mapping row names it" if mapping_given else "no mapping file was given")
            + "; nothing in the data says which scan it belongs to",
        }[emb_status]
        scan_id = emb_scans[0] if emb_status == "confirmed" else None
        return MappingRecord(
            image_id=image.image_id,
            scan_id=scan_id,
            station_id=index.scan_to_station.get(scan_id) if scan_id else None,
            status=emb_status,
            evidence_type="e57_associated_guid" if emb_value else "none",
            evidence_value=emb_value,
            reason=reason,
            candidate_scan_ids=emb_scans if emb_status == "ambiguous" else [],
            hints=hints,
        )

    # Several rows may name the same image. If they disagree, that is a conflict in the
    # mapping file itself and neither row wins.
    targets = {tuple(sorted(scans)) for _entry, scans, _hint in rows}
    entry, scans, hint = rows[0]
    if hint:
        hints.append(hint)
    if len(targets) > 1:
        return MappingRecord(
            image_id=image.image_id,
            status="conflict",
            evidence_type=entry.evidence_type,
            evidence_value="; ".join(f"{e.origin} -> {e.target}" for e, _s, _h in rows),
            reason=f"{len(rows)} mapping rows name different scans for this image; a mapping "
            "file that contradicts itself is not evidence",
            candidate_scan_ids=sorted({s for _e, ss, _h in rows for s in ss}),
            hints=hints,
        )

    if not scans:
        return MappingRecord(
            image_id=image.image_id,
            status="orphan",
            evidence_type=entry.evidence_type,
            evidence_value=str(entry.target),
            reason=f"{entry.origin} maps this image to {entry.target}, which this survey does "
            "not contain",
            hints=hints,
        )
    if len(scans) > 1:
        return MappingRecord(
            image_id=image.image_id,
            status="ambiguous",
            evidence_type=entry.evidence_type,
            evidence_value=str(entry.target),
            reason=f"{entry.origin} maps this image to {entry.target}, which names "
            f"{len(scans)} scans; first match is not a resolution",
            candidate_scan_ids=sorted(scans),
            hints=hints,
        )

    scan_id = scans[0]
    # The file's own association is machine evidence. A mapping file that contradicts it is a
    # conflict, never an override: whichever is wrong, we do not know which (§13).
    if emb_status in ("confirmed", "ambiguous") and scan_id not in emb_scans:
        return MappingRecord(
            image_id=image.image_id,
            status="conflict",
            evidence_type=entry.evidence_type,
            evidence_value=f"{entry.origin} -> {entry.target}; "
            f"E57 associatedData3DGuid {emb_value} -> {emb_scans}",
            reason=f"{entry.origin} maps this image to {scan_id}, but the E57 itself associates "
            f"it with {emb_scans}. A mapping file does not silently overwrite the file's own "
            "evidence; resolve the disagreement and re-run",
            candidate_scan_ids=sorted({scan_id, *emb_scans}),
            hints=hints,
        )
    if emb_status == "orphan":
        hints.append(
            f"the E57 associates this image with scan GUID {emb_value}, which is not in this "
            "file; the mapping file was used instead"
        )

    confirmed = entry.evidence_type == "vendor_manifest"
    agrees = emb_status == "confirmed" and scan_id in emb_scans
    return MappingRecord(
        image_id=image.image_id,
        scan_id=scan_id,
        station_id=index.scan_to_station.get(scan_id),
        status="confirmed" if (confirmed or agrees) else "manual",
        evidence_type="e57_associated_guid" if agrees else entry.evidence_type,
        evidence_value=emb_value if agrees else str(entry.target),
        reason=(
            f"{entry.origin} and the E57's own associatedData3DGuid agree on {scan_id}"
            if agrees
            else f"{entry.origin} maps this image to {entry.target}"
            + (
                " (a machine-generated vendor index)"
                if confirmed
                else " (user-provided; not machine-verifiable)"
            )
        ),
        hints=hints,
    )


# ---------------------------------------------------------------- public entry point


def build_mapping_report(
    path: str | Path,
    mapping: str | Path | None = None,
    vendor_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
    compute_hash: bool = True,
    inv: E57Inventory | None = None,
) -> PanoMappingReport:
    """Discover images, map them to scans by evidence, and report what was and was not mapped.

    ``images_dir`` discovers external image files instead of the E57's embedded ``/images2D``.
    An external file declares neither a projection nor an association, so without ``mapping``
    every one of them comes back ``unmapped`` — by design (§15).
    """
    from minegs.ingest.e57.inventory import inventory

    p = Path(path)
    inv = inv or inventory(p, compute_hash=False)

    if images_dir is not None:
        images = discover_external_images(images_dir)
    else:
        images = discover_embedded_images(p)

    entries: list[ExplicitEntry] = []
    if vendor_manifest is not None:
        entries.extend(read_vendor_manifest(vendor_manifest))
    if mapping is not None:
        entries.extend(read_mapping_file(mapping))

    records, unresolved, issues, notes = map_images(inv, images, entries)

    if not images:
        notes.append(
            f"no images discovered in {images_dir or p}; panoramas for this survey, if any, "
            "are somewhere else"
        )
    for im in images:
        for msg in im.issues:
            issues.append(f"{im.image_id}: {msg}")

    sha = sha256_file(p) if compute_hash else None
    return PanoMappingReport(
        source_file=str(p.resolve()),
        source_sha256=sha,
        hash_skipped_reason=None if compute_hash else "requested with --no-hash",
        image_root=str(Path(images_dir).resolve()) if images_dir is not None else None,
        scan_count=inv.scan_count,
        images=images,
        mappings=records,
        unresolved_references=unresolved,
        issues=issues,
        notes=notes,
        provenance=ProvenanceRecord(
            git_commit=git_commit(),
            source_assets=[
                SourceAsset(path=str(p.resolve()), sha256=sha or "", size_bytes=p.stat().st_size)
            ],
            tool_versions=tool_versions(),
        ),
    )
