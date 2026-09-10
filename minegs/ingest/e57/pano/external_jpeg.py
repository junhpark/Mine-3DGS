"""Panoramas delivered as separate JPEG/PNG files + a ``station_id,pano_id`` CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57.pano.base import PanoRecord, PanoSource, read_mapping


class ExternalJpeg(PanoSource):
    def __init__(
        self, root: str | Path, mapping: str | Path, convention: PanoConvention | None = None
    ) -> None:
        self.root = Path(root)
        self.mapping = read_mapping(mapping)
        self.convention = convention or PanoConvention(source="ExternalJpeg")
        self._records: list[PanoRecord] | None = None

    def _resolve(self, pano_id: str) -> Path:
        p = self.root / pano_id
        if p.exists():
            return p
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            if (self.root / (pano_id + ext)).exists():
                return self.root / (pano_id + ext)
        raise ContractError(f"panorama {pano_id!r} not found under {self.root}")

    def list_panoramas(self) -> list[PanoRecord]:
        if self._records is None:
            recs = []
            for sid, pid in self.mapping.items():
                p = self._resolve(pid)
                with Image.open(p) as im:
                    w, h = im.size
                recs.append(PanoRecord(sid, pid, w, h, str(p)))
            self._records = recs
        return self._records

    def load(self, pano_id: str) -> np.ndarray:
        with Image.open(self._resolve(pano_id)) as im:
            return np.asarray(im.convert("RGB"))
