from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError
from minegs.ingest.common.geometry import PanoConvention


@dataclass(frozen=True)
class PanoRecord:
    station_id: str
    pano_id: str
    width: int | None = None
    height: int | None = None
    path: str | None = None


class PanoSource(ABC):
    convention: PanoConvention = PanoConvention()

    @abstractmethod
    def list_panoramas(self) -> list[PanoRecord]: ...

    @abstractmethod
    def load(self, pano_id: str) -> np.ndarray:
        """Equirectangular image (H, W, 3) uint8."""

    def for_station(self, station_id: str) -> PanoRecord:
        for r in self.list_panoramas():
            if r.station_id == station_id:
                return r
        raise ContractError(f"no panorama mapped to station {station_id!r}")


def read_mapping(path: str | Path) -> dict[str, str]:
    """``station_id,pano_id`` CSV -> dict."""
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for row in rows:
        if not row or row[0].startswith("#") or row[0].strip().lower() == "station_id":
            continue
        out[row[0].strip()] = row[1].strip()
    if not out:
        raise ContractError(f"{path}: empty station<->pano mapping")
    return out


def get_pano_source(
    kind: str,
    root: str | Path | None = None,
    mapping: str | Path | None = None,
    convention: PanoConvention | None = None,
    e57_path: str | Path | None = None,
) -> PanoSource:
    conv = convention or PanoConvention()
    if kind == "E57Embedded":
        from minegs.ingest.e57.pano.e57_embedded import E57Embedded

        if e57_path is None:
            raise ContractError("E57Embedded needs e57_path")
        return E57Embedded(e57_path, conv, mapping)
    if kind == "ExternalJpeg":
        from minegs.ingest.e57.pano.external_jpeg import ExternalJpeg

        if root is None or mapping is None:
            raise ContractError("ExternalJpeg needs root and mapping")
        return ExternalJpeg(root, mapping, conv)
    if kind == "VendorExport":
        from minegs.ingest.e57.pano.vendor_export import VendorExport

        if root is None:
            raise ContractError("VendorExport needs root")
        return VendorExport(root, conv, mapping)
    raise ContractError(f"unknown PanoSource {kind!r}")
