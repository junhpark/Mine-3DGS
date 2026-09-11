"""Vendor exports (Leica Cyclone / FARO Scene / Trimble): a folder of panoramas plus the
vendor's own index naming each station. Extend ``_parse_index`` per vendor as they show up."""

from __future__ import annotations

import json
from pathlib import Path

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention
from minegs.ingest.e57.pano.base import read_mapping
from minegs.ingest.e57.pano.external_jpeg import ExternalJpeg


class VendorExport(ExternalJpeg):
    """A vendor panorama folder. The index is required, not optional.

    Earlier this class read the station out of the file name (``Station_03.jpg`` -> ``S03``).
    That is exactly the undocumented vendor convention the Phase 0B.2 mapping contract
    forbids as evidence (docs/ROADMAP.md §0B.2): the pattern holds until an export drops the
    prefix, renumbers from 1, or writes the stations out of order, and when it breaks every
    panorama is attributed to the wrong station silently. Pass the vendor's index, or write a
    mapping file — ``minegs ingest e57 pano-map`` prints what is available to map.
    """

    def __init__(
        self,
        root: str | Path,
        convention: PanoConvention | None = None,
        index: str | Path | None = None,
    ) -> None:
        root = Path(root)
        if index is None:
            raise ContractError(
                f"{root}: VendorExport needs the vendor's index (CSV/JSON naming each "
                "panorama's station). Station ids are not inferred from file names — that "
                "convention is undocumented, breaks silently, and mis-attributes every "
                "panorama when it does. Run `minegs ingest e57 pano-map <file.e57>` to see "
                "the images, then pass a station_id,image_name mapping."
            )
        mapping = self._parse_index(root, index)
        tmp = root / ".minegs_station_map.csv"
        tmp.write_text("station_id,pano_id\n" + "".join(f"{k},{v}\n" for k, v in mapping.items()))
        super().__init__(root, tmp, convention or PanoConvention(source="VendorExport"))

    @staticmethod
    def _parse_index(root: Path, index: str | Path) -> dict[str, str]:
        index = Path(index)
        if not index.is_file():
            raise ContractError(f"vendor index not found: {index}")
        if index.suffix.lower() == ".json":
            data = json.loads(index.read_text())
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            return {str(d["station_id"]): str(d["pano_id"]) for d in data}
        return read_mapping(index)
