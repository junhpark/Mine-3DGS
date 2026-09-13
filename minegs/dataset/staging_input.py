"""Reader for the Phase 0B.3 staging tree — 0C's only input (§3).

0C does not reinterpret the E57. It reads the three artifacts 0B wrote (``inventory.json``,
``pano_mapping.json``, ``extraction_manifest.json``) through their own pydantic contracts,
and the files they name. Anything the artifacts do not say, 0C does not know.

Checks that make a tree usable:

* the three artifacts describe the same bytes (one ``source_sha256``), and it is present —
  a tree extracted with ``--no-hash`` cannot anchor a dataset's provenance (§21);
* the extraction is ``registered`` in the ``SOURCE`` frame, because the frame chain starts
  there. A ``--raw`` (SCANNER-frame) tree is refused, not re-registered here;
* every file the manifests reference exists under the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minegs.core.errors import ContractError
from minegs.core.frames import SE3
from minegs.core.provenance import SourceAsset, sha256_file
from minegs.ingest.e57.extract import E57ExtractionManifest, ImageOutput, ScanOutput
from minegs.ingest.e57.images import ImageAsset
from minegs.ingest.e57.mapping import MappingRecord, PanoMappingReport
from minegs.ingest.e57.models import E57Inventory

ARTIFACTS = ("inventory.json", "pano_mapping.json", "extraction_manifest.json")


@dataclass
class StagingTree:
    root: Path
    inventory: E57Inventory
    mapping: PanoMappingReport
    extraction: E57ExtractionManifest
    artifact_assets: list[SourceAsset]

    # ---------------------------------------------------------------- lookups
    def scan_output(self, scan_id: str) -> ScanOutput:
        try:
            return self.extraction.scan_output(scan_id)
        except KeyError:
            raise ContractError(f"{self.root}: scan {scan_id!r} was not extracted") from None

    def scan_ply(self, scan_id: str) -> Path:
        # Root-relative, not the absolute path the extractor recorded: a staging tree that was
        # moved is still the same tree, and the recorded path then points at nothing.
        p = self.root / "scans" / Path(self.scan_output(scan_id).path).name
        if not p.is_file():
            raise ContractError(f"{self.root}: missing scan file {p.name}")
        return p

    def scan_pose(self, scan_id: str) -> SE3:
        out = self.scan_output(scan_id)
        if out.pose is None:
            raise ContractError(
                f"{scan_id}: no pose recorded in the extraction manifest; a registered "
                "extraction always carries one"
            )
        return out.pose.se3()

    def image_output(self, image_id: str) -> ImageOutput:
        for im in self.extraction.image_outputs:
            if im.image_id == image_id:
                return im
        raise ContractError(f"{self.root}: image {image_id!r} was not written by the extractor")

    def image_path(self, image_id: str) -> Path:
        out = self.image_output(image_id)
        p = self.root / "images" / Path(out.path).name if out.extracted else Path(out.path)
        if not p.is_file():
            raise ContractError(f"{self.root}: missing image file {p}")
        return p

    def asset(self, image_id: str) -> ImageAsset:
        for a in self.mapping.images:
            if a.image_id == image_id:
                return a
        raise ContractError(f"{self.root}: image {image_id!r} is not in pano_mapping.json")

    def record(self, image_id: str) -> MappingRecord:
        try:
            return self.mapping.record_for(image_id)
        except KeyError:
            raise ContractError(f"{self.root}: no mapping record for {image_id!r}") from None

    def station_of(self, scan_id: str) -> str:
        for st in self.inventory.station_candidates:
            if scan_id in st.scan_ids:
                return st.station_id
        raise ContractError(f"{self.root}: scan {scan_id!r} belongs to no station candidate")

    @property
    def source_sha256(self) -> str:
        assert self.extraction.source_sha256 is not None  # checked in load_staging
        return self.extraction.source_sha256


def load_staging(path: str | Path) -> StagingTree:
    root = Path(path)
    if not root.is_dir():
        raise ContractError(f"{root}: staging directory not found")
    for name in ARTIFACTS:
        if not (root / name).is_file():
            raise ContractError(
                f"{root}: missing {name}. The input to a dataset build is a Phase 0B.3 staging "
                "tree written by `minegs ingest e57 extract`, not a dataset/ or a raw E57."
            )
    inventory = E57Inventory.load(root / "inventory.json")
    mapping = PanoMappingReport.load(root / "pano_mapping.json")
    extraction = E57ExtractionManifest.load(root / "extraction_manifest.json")

    if extraction.registration != "registered" or extraction.output_frame != "SOURCE":
        raise ContractError(
            f"{root}: extraction is {extraction.registration} in the {extraction.output_frame} "
            "frame. A dataset needs scans registered in the E57's SOURCE frame (extract without "
            "--raw); scanner-frame clouds cannot be placed without the pose this tree lacks."
        )
    digests = {
        "inventory.json": inventory.file.sha256,
        "pano_mapping.json": mapping.source_sha256,
        "extraction_manifest.json": extraction.source_sha256,
    }
    if any(d is None for d in digests.values()):
        missing = [k for k, v in digests.items() if v is None]
        raise ContractError(
            f"{root}: {missing} carry no source SHA-256 (extracted with --no-hash). A dataset "
            "must be able to name the bytes it came from (§21); re-run the extraction with "
            "hashing."
        )
    if len(set(digests.values())) != 1:
        raise ContractError(
            f"{root}: the three artifacts disagree about the source digest: {digests}"
        )
    for s in extraction.scan_outputs:
        if s.sha256 is None:
            raise ContractError(f"{root}: scan {s.scan_id} has no recorded digest (§21)")
    assets = [
        SourceAsset(
            path=str(root / name),
            sha256=sha256_file(root / name),
            size_bytes=(root / name).stat().st_size,
        )
        for name in ARTIFACTS
    ]
    tree = StagingTree(root, inventory, mapping, extraction, assets)
    for s in extraction.scan_outputs:
        tree.scan_ply(s.scan_id)  # existence
    return tree
