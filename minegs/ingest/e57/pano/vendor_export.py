"""Vendor exports (Leica Cyclone / FARO Scene / Trimble): a folder of panoramas whose file
names encode the station, or a vendor CSV/JSON index. Handles the name-based case; extend
``_parse_index`` per vendor as they show up."""

from __future__ import annotations

import json
import re
from pathlib import Path

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57.pano.base import read_mapping
from minegs.ingest.e57.pano.external_jpeg import ExternalJpeg

_STATION_RE = re.compile(r"(?:Station|Setup|Scan|S)[_\- ]?(\d+)", re.IGNORECASE)


class VendorExport(ExternalJpeg):
    def __init__(
        self,
        root: str | Path,
        convention: PanoConvention | None = None,
        index: str | Path | None = None,
    ) -> None:
        root = Path(root)
        mapping = self._parse_index(root, index) if index else self._from_names(root)
        tmp = root / ".minegs_station_map.csv"
        tmp.write_text("station_id,pano_id\n" + "".join(f"{k},{v}\n" for k, v in mapping.items()))
        super().__init__(root, tmp, convention or PanoConvention(source="VendorExport"))

    @staticmethod
    def _from_names(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(root.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                continue
            m = _STATION_RE.search(p.stem)
            if m:
                out[f"S{int(m.group(1)):02d}"] = p.name
        if not out:
            raise ContractError(f"{root}: could not infer station ids from panorama file names")
        return out

    @staticmethod
    def _parse_index(root: Path, index: str | Path) -> dict[str, str]:
        index = Path(index)
        if index.suffix.lower() == ".json":
            data = json.loads(index.read_text())
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            return {str(d["station_id"]): str(d["pano_id"]) for d in data}
        return read_mapping(index)
