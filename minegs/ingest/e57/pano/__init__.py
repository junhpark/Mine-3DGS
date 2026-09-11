"""PanoSource adapters (§6.1). The only contract is ``station_id <-> panorama_id``;
E57 is *not* assumed to embed panoramas."""

from minegs.ingest.e57.pano.base import PanoRecord, PanoSource, get_pano_source

__all__ = ["PanoRecord", "PanoSource", "get_pano_source"]
