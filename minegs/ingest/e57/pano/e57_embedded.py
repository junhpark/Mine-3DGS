"""Panoramas embedded in the E57 (``/images2D`` spherical representation, JPEG blob)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57 import _nodes
from minegs.ingest.e57.inventory import inventory, list_images2d
from minegs.ingest.e57.pano.base import PanoRecord, PanoSource, read_mapping


class E57Embedded(PanoSource):
    def __init__(
        self,
        e57_path: str | Path,
        convention: PanoConvention | None = None,
        mapping: str | Path | None = None,
    ) -> None:
        self.path = Path(e57_path)
        self.convention = convention or PanoConvention(source="E57Embedded")
        self.inv = inventory(self.path, compute_hash=False)
        self.images = list_images2d(self.path)
        self.mapping = read_mapping(mapping) if mapping else None

    def list_panoramas(self) -> list[PanoRecord]:
        """Station -> panorama records.

        Requires an explicit ``station_id,pano_id`` mapping. Phase 0B.1 deliberately stops at
        *detecting* images: inferring which panorama belongs to which station from GUIDs or
        file order is the Phase 0B.2 contract, and guessing it here would be exactly the
        vendor assumption the ingest path must not make.
        """
        if not self.images:
            raise ContractError(
                f"{self.path}: no embedded images2D entries; use ExternalJpeg/VendorExport"
            )
        if not self.mapping:
            raise ContractError(
                f"{self.path}: an explicit station_id,pano_id mapping is required. Automatic "
                "station/panorama mapping is Phase 0B.2 (docs/ROADMAP.md); run "
                "`minegs ingest e57 inventory` to see the available scans and images."
            )
        by_name = {im["name"]: im for im in self.images if im.get("name")}
        by_idx = {str(im["index"]): im for im in self.images}
        recs = []
        for sid, pid in self.mapping.items():
            im = by_name.get(pid) or by_idx.get(pid)
            if im is None:
                raise ContractError(f"pano {pid!r} for station {sid!r} not in E57")
            recs.append(PanoRecord(sid, str(im["index"]), im.get("width"), im.get("height")))
        return recs

    def load(self, pano_id: str) -> np.ndarray:
        with _nodes.open_e57(self.path) as e57:
            node = e57.image_file.root()["images2D"].get(int(pano_id))
            rep = node["sphericalRepresentation"]
            blob = rep["jpegImage"] if rep.isDefined("jpegImage") else rep["pngImage"]
            buf = np.empty(blob.byteCount(), dtype=np.uint8)
            blob.read(buf, 0, blob.byteCount())
        with Image.open(io.BytesIO(buf.tobytes())) as im:
            return np.asarray(im.convert("RGB"))
