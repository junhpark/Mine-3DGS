"""Panoramas embedded in the E57 (``/images2D`` spherical representation, JPEG blob)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57.inventory import _pye57, inventory
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
        self.inv = inventory(self.path)
        self.mapping = read_mapping(mapping) if mapping else None

    def list_panoramas(self) -> list[PanoRecord]:
        recs = []
        if self.mapping:
            by_name = {im.name: im for im in self.inv.images}
            by_idx = {str(im.index): im for im in self.inv.images}
            for sid, pid in self.mapping.items():
                im = by_name.get(pid) or by_idx.get(pid)
                if im is None:
                    raise ContractError(f"pano {pid!r} for station {sid!r} not in E57")
                recs.append(PanoRecord(sid, str(im.index), im.width, im.height))
            return recs
        for s_idx, im_idx in self.inv.station_pano_map().items():
            if im_idx is not None:
                im = self.inv.images[im_idx]
                recs.append(PanoRecord(f"S{s_idx + 1:02d}", str(im.index), im.width, im.height))
        if not recs:
            raise ContractError(
                f"{self.path}: no embedded panoramas; use ExternalJpeg/VendorExport"
            )
        return recs

    def load(self, pano_id: str) -> np.ndarray:
        pye57 = _pye57()
        e57 = pye57.E57(str(self.path))
        try:
            node = e57.image_file.root()["images2D"].get(int(pano_id))
            rep = node["sphericalRepresentation"]
            blob = rep["jpegImage"] if rep.isDefined("jpegImage") else rep["pngImage"]
            buf = np.empty(blob.byteCount(), dtype=np.uint8)
            blob.read(buf, 0, blob.byteCount())
        finally:
            e57.close()
        with Image.open(io.BytesIO(buf.tobytes())) as im:
            return np.asarray(im.convert("RGB"))
